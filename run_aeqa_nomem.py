import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# OpenAI API 키 설정
from src.const import OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import logging
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
import habitat_sim

from ultralytics import SAM, YOLOWorld

from src.habitat import (
    pose_habitat_to_tsdf, pos_habitat_to_normal
)
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner
from src.scene_aeqa import Scene
from src.utils import (
    calc_agent_subtask_distance, get_pts_angle_aeqa,
)
from src.logger_aeqa import Logger
from src.siglip_itm import SigLipITM
from src.descriptor_extractor import DescriptorExtractor, DescriptorExtractorConfig
from src.data_models import Frontier


class ElapsedTimeFormatter(logging.Formatter):
    """Formats log time to be elapsed time from the start."""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed_seconds = record.created - self.start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


@dataclass
class AppState:
    """모든 주요 컴포넌트와 설정을 담는 데이터 클래스."""
    cfg: OmegaConf
    device: torch.device
    logger: Logger
    itm: SigLipITM
    desc_extractor: DescriptorExtractor
    detection_model: YOLOWorld
    sam_predictor: SAM
    cam_intr: np.ndarray
    min_depth: Optional[float]
    max_depth: Optional[float]
    # NOTE: observation_memory 제거 - 메모리 없이 수행


def setup(cfg_file: str, start_ratio: float, end_ratio: float) -> AppState:
    """설정 로딩, 로깅, 모델 초기화 등 모든 준비 작업을 수행합니다."""
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if os.path.abspath(cfg_file) != os.path.abspath(os.path.join(cfg.output_dir, os.path.basename(cfg_file))):
        os.system(f"cp {cfg_file} {cfg.output_dir}")

    # 로깅 설정
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    root_logger = logging.getLogger()
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.INFO)

    # 1. 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 2. 메인 로그 파일 핸들러 (초기 설정 및 최종 결과용)
    logging_path = os.path.join(str(cfg.output_dir), f"log_{start_ratio:.2f}_{end_ratio:.2f}.log")
    main_file_handler = logging.FileHandler(logging_path, mode="w")
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    detection_model = YOLOWorld(cfg.yolo_model_name)
    sam_predictor = SAM(cfg.sam_model_name)
    itm = SigLipITM(
        device=device,
        model_name=cfg.siglip_model_name,
        pretrained=cfg.siglip_pretrained,
        backend=cfg.siglip_backend,
    )
    desc_extractor_config = DescriptorExtractorConfig(
        use_chain_descriptors=cfg.use_chain_descriptors,
        gpt_model=cfg.gpt_model,
        n_descriptors=cfg.descriptors_per_frontier,
    )
    desc_extractor = DescriptorExtractor(itm, desc_extractor_config)
    # AEQA logger는 n_total_questions가 필요하지만, 여기서는 임시로 1로 설정
    # 실제로는 run_evaluation에서 올바른 값으로 설정됨
    logger = Logger(cfg.output_dir, start_ratio, end_ratio, 1, voxel_size=cfg.tsdf_grid_size)

    cam_intr = get_cam_intr(cfg.hfov, cfg.img_height, cfg.img_width)
    min_depth = cfg.min_depth if hasattr(cfg, "min_depth") else None
    max_depth = cfg.max_depth if hasattr(cfg, "max_depth") else None

    logging.info("Setup complete.")
    
    return AppState(
        cfg=cfg, device=device, logger=logger, itm=itm,
        desc_extractor=desc_extractor, detection_model=detection_model,
        sam_predictor=sam_predictor, cam_intr=cam_intr, min_depth=min_depth, max_depth=max_depth,
        # NOTE: observation_memory 제거
    )


def run_evaluation(app_state: AppState, start_ratio: float, end_ratio: float, question_id: Optional[str] = None):
    """준비된 상태를 바탕으로 AEQA 데이터셋을 순회하며 평가를 실행합니다."""
    cfg = app_state.cfg
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load AEQA dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    
    # Filter by specific question_id if provided
    if question_id:
        target_question = None
        for q in questions_list:
            if q["question_id"] == question_id:
                target_question = q
                break
        
        if target_question:
            questions_list = [target_question]
            logging.info(f"Running for single question: {question_id}")
        else:
            logging.error(f"Question ID {question_id} not found in {cfg.questions_list_path}. Exiting.")
            return
    else:
        # only process a subset of the questions
        questions_list = questions_list[
            int(start_ratio * total_questions) : int(end_ratio * total_questions)
        ]
        logging.info(f"Number of questions after splitting: {len(questions_list)}")
    
    logging.info(f"Question path: {cfg.questions_list_path}")
    
    # AEQA logger를 올바른 total_questions로 재초기화
    app_state.logger = Logger(cfg.output_dir, start_ratio, end_ratio, len(questions_list), voxel_size=cfg.tsdf_grid_size)

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    
    root_logger = logging.getLogger()
    # 메인 파일 핸들러를 찾아서 evaluation 중에는 잠시 제거했다가 끝나고 다시 추가합니다.
    main_file_handler = None
    for handler in root_logger.handlers:
        # 'log_'로 시작하는 파일명을 가진 핸들러를 메인 핸들러로 간주합니다.
        if isinstance(handler, logging.FileHandler) and "log_" in os.path.basename(handler.baseFilename):
             main_file_handler = handler
             break

    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        
        # 이미 처리된 질문은 건너뜀
        if question_id in app_state.logger.success_list or question_id in app_state.logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
            
        question_file_handler = None
        try:
            # --- Set up question-specific file logging ---
            if main_file_handler:
                root_logger.removeHandler(main_file_handler)

            # 질문 결과 폴더 내에 로그 파일 생성
            log_dir = os.path.join(app_state.cfg.output_dir, question_id)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{question_id}.log")
            question_file_handler = logging.FileHandler(log_path, mode="w")
            question_file_handler.setFormatter(formatter)
            root_logger.addHandler(question_file_handler)

            logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")
            logging.info(f"Question: {question_data['question']}")
            logging.info(f"Answer: {question_data['answer']}")

            question = question_data["question"]
            answer = question_data["answer"]
            pts, angle = get_pts_angle_aeqa(
                question_data["position"], question_data["rotation"]
            )

            # load scene
            scene = Scene(
                scene_id, cfg, cfg_cg, app_state.detection_model, app_state.sam_predictor,
                vlm_model=app_state.itm, device=app_state.device,
            )

            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            max_steps = max(int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio), 50)

            # NOTE: 매 question마다 새로운 TSDF planner 생성 (frontier 완전 초기화)
            logging.info(f"Creating new TSDF planner for question {question_id} (frontier reset)")
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds, voxel_size=cfg.tsdf_grid_size, floor_height=floor_height,
                floor_height_offset=0, pts_init=pts, init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

            episode_context = {
                "start_position": pts, "floor_height": floor_height, "tsdf_bounds": tsdf_bnds,
                "visited_positions": [], "observations_history": [], "step_count": 0,
                "question_observation_cache": {}, # Initialize short-term cache
                "question_id": question_id, # Initialize question_id
                "question_str": question, # Initialize question_str
                "target_objects": [], # Initialize target_objects
            }

            episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = app_state.logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],
            )

            episode_context["tsdf_planner"] = tsdf_planner
            episode_context["eps_frontier_dir"] = eps_frontier_dir  # Store for later use

            # 질문 메타데이터 생성
            question_metadata = {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "goal": question,  # AEQA에서는 질문 자체가 목표
                "goal_type": "description",  # AEQA는 항상 description 타입
                "viewpoints": [pts],  # 시작 위치를 viewpoint로 설정
                "gt_subtask_explore_dist": 0,  # AEQA에는 ground truth 탐색 거리가 없음
            }

            logging.info(f"\nQuestion {question_idx + 1}/{len(questions_list)}: {question}")
            
            task_result = run_question(
                app_state=app_state,
                question_id=question_id,
                question_metadata=question_metadata,
                scene=scene,
                episode_context=episode_context,
                pts=pts,
                angle=angle,
                max_steps=max_steps,
                tsdf_planner=tsdf_planner,
            )
            
            app_state.logger.save_results()
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

            # scene 정리
            scene.close()
            del scene

        finally:
            # --- 질문 로깅 핸들러 정리 및 메인 핸들러 복원 ---
            if question_file_handler:
                question_file_handler.close()
                root_logger.removeHandler(question_file_handler)
            if main_file_handler:
                root_logger.addHandler(main_file_handler)

    app_state.logger.save_results()
    app_state.logger.aggregate_results()


def run_question(
    app_state: AppState,
    question_id: str,
    question_metadata: Dict[str, Any],
    scene: Scene,
    episode_context: Dict[str, Any],
    pts: np.ndarray,
    angle: float,
    max_steps: int,
    tsdf_planner: TSDFPlanner,
):
    """
    하나의 질문을 자율적으로 수행합니다.
    NOTE: 메모리 없는 버전 - 매 question마다 독립적으로 탐색
    """
    # Initialize the observation history for this specific question
    episode_context["question_full_observation_history"] = {}
    episode_context["frontier_score_cache"] = {} # Initialize score cache for this question

    question = question_metadata.get("question")
    answer = question_metadata.get("answer")
    
    # 초기 상태 설정
    current_pts = pts.copy()
    current_angle = angle
    task_success = False
    gpt_answer = None
    confidence_score = 0.0

    # --- Phase 1: Goal Pre-processing ---
    target_objects = []
    logging.info(f"Goal is a description. Extracting target objects...")
    main_targets, _context_objects = app_state.desc_extractor.extract_target_objects_from_description(question)
    target_objects = main_targets
    
    logging.info(f"Refined Target Objects: {target_objects}")

    # Store refined targets and original question string
    episode_context["target_objects"] = target_objects if isinstance(target_objects, list) else []
    episode_context["question_str"] = question
    episode_context["accumulated_descriptors"] = set()  # Use set for automatic deduplication

    # run steps
    cnt_step = 0
    while cnt_step < max_steps:
        cnt_step += 1
        
        # Initialize per-step variables
        candidates_info = None
        chosen_target = None
        best_descriptors = []

        # --- Phase 2: 계층적 전략 수행 ---
        # 목표 지속성(Goal Persistence)을 적용합니다.
        if tsdf_planner.max_point is None:
            logging.info("No long-term target. Stop, Scan, and Select.")
            
            # --- 1. 관측 및 맵 업데이트 ---
            current_step_observations = _observe_and_update_maps(
                scene=scene, tsdf_planner=tsdf_planner, current_pts=current_pts,
                current_angle=current_angle, cnt_step=cnt_step, cfg=app_state.cfg,
                cam_intr=app_state.cam_intr, min_depth=app_state.min_depth,
                max_depth=app_state.max_depth, eps_frontier_dir=episode_context["eps_frontier_dir"],
            )
            
            for obs in current_step_observations:
                episode_context["question_full_observation_history"][obs['name']] = obs

            for frontier in tsdf_planner.frontiers:
                if frontier.source_observation_name is None:
                    relevant_obs_list = get_relevant_observations(
                        frontier, current_step_observations,
                        app_state.cfg.relevant_view_angle_threshold_deg
                    )
                    if len(relevant_obs_list) > 0:
                        frontier.source_observation_name = relevant_obs_list[0]['name']
                        frontier.cam_pose = relevant_obs_list[0]['cam_pose']
                        frontier.depth = relevant_obs_list[0]['depth']
            
            # --- 2. 캐시 업데이트 ---
            active_frontier_ids = {f.frontier_id for f in tsdf_planner.frontiers}
            required_obs_names = {f.source_observation_name for f in tsdf_planner.frontiers if f.source_observation_name}
            episode_context["question_observation_cache"] = {
                name: obs for name, obs in episode_context["question_full_observation_history"].items() if name in required_obs_names
            }
            current_cache = episode_context.get("frontier_score_cache", {})
            episode_context["frontier_score_cache"] = {
                fid: score_data for fid, score_data in current_cache.items() if fid in active_frontier_ids
            }

            # Fallback: force-link frontier to nearest observation if none linked
            if not required_obs_names and tsdf_planner.frontiers:
                if episode_context["question_full_observation_history"]:
                    any_obs = next(iter(episode_context["question_full_observation_history"].values()))
                    for frontier in tsdf_planner.frontiers:
                        if frontier.source_observation_name is None:
                            frontier.source_observation_name = any_obs['name']
                            frontier.cam_pose = any_obs['cam_pose']
                            frontier.depth = any_obs['depth']
                    episode_context["question_observation_cache"][any_obs['name']] = any_obs

            # --- 3. 새로운 목표 선택 (메모리 없이) ---
            selection_dir = os.path.join(episode_context["eps_frontier_dir"], "..", "selection")
            os.makedirs(selection_dir, exist_ok=True)
            chosen_target, candidates_info, best_descriptors = _select_next_target(
                app_state=app_state, episode_context=episode_context,
                current_pts=current_pts,
                goal=question,
                goal_type="description",
                original_goal_str=question,
                tsdf_planner=tsdf_planner,
                selection_dir=selection_dir,
                target_objects=target_objects,
            )
            
            # 선택된 frontier의 descriptor를 누적 (QA를 위해)
            if best_descriptors and len(best_descriptors) > 0:
                episode_context["accumulated_descriptors"].update(best_descriptors)
                logging.info(f"Accumulated {len(episode_context['accumulated_descriptors'])} unique descriptors so far")
            
            if not chosen_target:
                logging.warning("Failed to select a new target frontier. Will re-scan next step.")
            
            if chosen_target:
                set_ok = tsdf_planner.set_next_navigation_point(
                    choice=chosen_target,
                    pts=current_pts,
                    objects=scene.objects,
                    cfg=app_state.cfg.planner,
                    pathfinder=scene.pathfinder
                )
                if not set_ok:
                    logging.warning("Failed to set navigation point for newly chosen target. Will retry next step.")
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

        elif tsdf_planner.target_point is None:
            logging.info(f"Committed target exists. Calculating next waypoint.")
            set_ok = tsdf_planner.set_next_navigation_point(
                choice=tsdf_planner.max_point,
                pts=current_pts,
                objects=scene.objects,
                cfg=app_state.cfg.planner,
                pathfinder=scene.pathfinder
            )
            if not set_ok:
                logging.warning("Failed to find path to committed target. Clearing for re-evaluation.")
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
        
        else:
            logging.info(f"Continuing towards current target: {tsdf_planner.target_point}")

        target_info = f"| Target: {tsdf_planner.target_point}" if tsdf_planner.target_point is not None else ""
        logging.info(f"\nStep {cnt_step}/{max_steps} {target_info}")

        # --- 4. 에이전트 한 스텝 이동 ---
        step_vals = tsdf_planner.agent_step(
            pts=current_pts,
            angle=current_angle,
            objects=scene.objects,
            snapshots=scene.snapshots,
            pathfinder=scene.pathfinder,
            cfg=app_state.cfg.planner if hasattr(app_state.cfg, "planner") else app_state.cfg,
            save_visualization=app_state.cfg.save_visualization,
        )

        if step_vals[0] is None:
            logging.warning("Agent step failed. Clearing targets to force re-evaluation.")
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            continue

        current_pts, current_angle, _, fig, _, waypoint_arrived = step_vals
        app_state.logger.log_step(pts_voxel=tsdf_planner.habitat2voxel(current_pts)[:2])
        episode_context["visited_positions"].append(current_pts.copy())
        
        if waypoint_arrived:
            logging.info("Intermediate waypoint reached.")
            tsdf_planner.target_point = None

            if tsdf_planner.max_point is not None:
                current_voxel_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))
                dist_to_max_point_m = np.linalg.norm(current_voxel_pos[:2] - tsdf_planner.max_point.position) * tsdf_planner._voxel_size
                if dist_to_max_point_m < 0.5: 
                    logging.info("Vicinity of long-term target reached. Clearing for full re-evaluation.")
                    tsdf_planner.max_point = None
            
        # --- Visualization (optional) ---
        if app_state.cfg.save_visualization and fig is not None:
            # A-EQA Logger uses separate methods (unlike GOAT-Bench's unified method)
            app_state.logger.save_topdown_visualization(
                cnt_step=cnt_step,
                fig=fig,
            )
            
            # Save frontier visualization if we have a target
            if tsdf_planner.max_point is not None and len(tsdf_planner.frontiers) > 0:
                app_state.logger.save_frontier_visualization(
                    cnt_step=cnt_step,
                    tsdf_planner=tsdf_planner,
                    max_point_choice=tsdf_planner.max_point,
                    global_caption=f"Question: {question}\nAnswer: {answer}"
                )

        # --- A-EQA: 답변 생성 시도 ---
        # Trigger condition: Target object found (high ITM score)
        should_attempt_answer = False
        
        # Check if we just found a highly relevant frontier (target object likely found)
        if chosen_target and len(episode_context["accumulated_descriptors"]) >= 5:
            recent_cache = episode_context.get("frontier_score_cache", {})
            if chosen_target.frontier_id in recent_cache:
                recent_itm_score = recent_cache[chosen_target.frontier_id].get("score_itm", 0.0)
                qa_trigger_threshold = app_state.cfg.get("qa_trigger_itm_threshold", 0.6)
                if recent_itm_score >= qa_trigger_threshold:
                    should_attempt_answer = True
                    logging.info(
                        f"Target object likely found! ITM score: {recent_itm_score:.3f} >= {qa_trigger_threshold}. "
                        f"Attempting to answer with {len(episode_context['accumulated_descriptors'])} descriptors."
                    )
        
        if should_attempt_answer:
            # Descriptor 기반으로 답변 생성 시도
            logging.info(f"Generating answer with {len(episode_context['accumulated_descriptors'])} unique descriptors")
            gpt_answer, confidence_score = _generate_answer_from_descriptors(
                app_state=app_state,
                question=question,
                accumulated_descriptors=episode_context["accumulated_descriptors"],
            )
            
            # A-EQA: 높은 신뢰도의 답변이 생성되면 탐색 종료
            # (실제 정확도는 나중에 ground truth와 비교하여 평가)
            if confidence_score >= app_state.cfg.get("qa_confidence_threshold", 0.7):
                logging.info(
                    f"High confidence answer generated at step {cnt_step} "
                    f"(confidence: {confidence_score:.2f}). Answer: '{gpt_answer}'"
                )
                logging.info(f"Ground truth: '{answer}'")
                
                # Answer accuracy check (A-EQA의 진짜 평가 기준)
                task_success = _check_answer_correctness(gpt_answer, answer)
                
                if task_success:
                    logging.info(f"✓ SUCCESS: Answer is CORRECT!")
                else:
                    logging.info(f"✗ FAIL: Answer is INCORRECT (but high confidence was reached)")
                break
            
        # Check for max steps
        if cnt_step >= max_steps:
            logging.warning(f"Max steps reached ({max_steps}). Ending question.")
            break
    
    # Max steps 도달 시 마지막 답변 시도
    if not task_success and len(episode_context["accumulated_descriptors"]) > 0:
        logging.info("Max steps reached. Attempting final answer generation...")
        gpt_answer, confidence_score = _generate_answer_from_descriptors(
            app_state=app_state,
            question=question,
            accumulated_descriptors=episode_context["accumulated_descriptors"],
        )
        # 낮은 confidence여도 답변이 있으면 기록하고 정확도 체크
        if gpt_answer:
            logging.info(f"Final answer (confidence: {confidence_score:.2f}): '{gpt_answer}'")
            logging.info(f"Ground truth: '{answer}'")
            task_success = _check_answer_correctness(gpt_answer, answer)
            if task_success:
                logging.info(f"✓ SUCCESS: Final answer is CORRECT!")
            else:
                logging.info(f"✗ FAIL: Final answer is INCORRECT")
    
    # 결과 로깅
    logging.info(f"\n{'='*60}")
    logging.info(f"A-EQA Episode Result:")
    logging.info(f"Question: {question}")
    logging.info(f"Ground Truth: {answer}")
    logging.info(f"Generated Answer: {gpt_answer if gpt_answer else 'No answer'}")
    logging.info(f"Steps taken: {cnt_step}")
    logging.info(f"Descriptors collected: {len(episode_context['accumulated_descriptors'])}")
    
    if task_success:
        logging.info(f"Result: ✓ SUCCESS (Answer is CORRECT)")
    else:
        if gpt_answer:
            logging.info(f"Result: ✗ FAIL (Answer is INCORRECT)")
        else:
            logging.info(f"Result: ✗ FAIL (No answer generated)")
    logging.info(f"{'='*60}\n")
    
    app_state.logger.log_episode_result(
        success=task_success,
        question_id=question_id,
        explore_dist=app_state.logger.explore_dist,
        gpt_answer=gpt_answer,
        n_filtered_snapshots=0,
        n_total_snapshots=0,
        n_total_frames=cnt_step,
    )
    
    return {
        "success": task_success,
        "final_position": current_pts,
        "final_angle": current_angle,
        "steps_taken": cnt_step,
        "answer": gpt_answer,
        "confidence": confidence_score,
    }


def _observe_and_update_maps(
    scene, tsdf_planner, current_pts, current_angle, cnt_step,
    cfg, cam_intr, min_depth, max_depth, eps_frontier_dir
) -> List[Dict[str, Any]]:
    """
    Performs a full 360-degree scan to observe the surroundings, then updates
    the TSDF and frontier maps.
    """
    num_scan_views = cfg.num_scan_views
    angle_increment = np.deg2rad(360 / num_scan_views)
    
    all_angles = [
        current_angle + (i * angle_increment)
        for i in range(num_scan_views)
    ]

    current_map_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))

    current_step_observations = []
    for view_idx, ang in enumerate(all_angles):
        obs, cam_pose = scene.get_observation(current_pts, angle=ang)
        rgb = obs["color_sensor"]
        depth_unfiltered = obs["depth_sensor"]

        obs_file_name = f"{cnt_step}-{view_idx}.png"
        obs_data = {
            "rgb": rgb, "angle": ang, "cam_pose": cam_pose, "name": obs_file_name, "depth": depth_unfiltered,
            "map_pos": (int(current_map_pos[0]), int(current_map_pos[1]))
        }
        current_step_observations.append(obs_data)

        if min_depth is not None and max_depth is not None:
            depth = np.where(
                (depth_unfiltered >= min_depth) & (depth_unfiltered <= max_depth),
                depth_unfiltered, np.nan,
            )
        else:
            depth = depth_unfiltered

        tsdf_planner.integrate(
            color_im=rgb, depth_im=depth, cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose), obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )

    # Update frontier map
    tsdf_planner.update_frontier_map(
        pts=current_pts,
        cfg=cfg.planner if hasattr(cfg, "planner") else cfg,
        scene=scene,
        cnt_step=cnt_step,
        save_frontier_image=cfg.save_visualization,
        eps_frontier_dir=eps_frontier_dir,
    )

    return current_step_observations


def _select_next_target(
    app_state: AppState,
    episode_context: Dict[str, Any],
    current_pts: np.ndarray,
    goal: Any,
    goal_type: str,
    original_goal_str: str,
    tsdf_planner: TSDFPlanner,
    selection_dir: str,
    target_objects: List[str],
) -> Tuple[Optional[Frontier], Optional[Dict[str, Any]], List[str]]:
    """
    Selects the next target using the conditional scoring model.
    Returns: (chosen_frontier, candidates_info, best_descriptors)
    """
    cfg = app_state.cfg
    score_cache = episode_context.get("frontier_score_cache", {})
    
    # Gather all necessary info for all frontiers
    frontiers_with_data = []
    question_obs_cache = episode_context.get("question_observation_cache", {})

    for f in tsdf_planner.frontiers:
        if f.source_observation_name and f.source_observation_name in question_obs_cache:
            source_obs = question_obs_cache[f.source_observation_name]
            frontiers_with_data.append({
                "frontier": f,
                "rgb": source_obs["rgb"],
                "key_objects": [] 
            })

    if not frontiers_with_data:
        logging.info("No frontiers with source observations found for scoring.")
        return None, None, []

    # Evaluate each frontier
    scored_frontiers = []
    cur_voxel = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))
    
    for data in frontiers_with_data:
        f = data["frontier"]
        
        # Get scores from cache or calculate them
        if f.frontier_id not in score_cache:
            logging.info(f"Cache MISS for Frontier {f.frontier_id}. Performing VLM analysis.")
            
            descriptors, vlm_likelihood_score, _ = app_state.desc_extractor.analyze_scene_for_goal(
                rgb=data["rgb"],
                goal=goal,
                key_objects=data["key_objects"]
            )
            
            # ITM score calculation
            if descriptors and target_objects:
                all_itm_scores = []
                for target_obj in target_objects:
                    itm_scores = app_state.itm.text_text_scores(target_obj, descriptors)
                    if itm_scores is not None and itm_scores.size > 0:
                        all_itm_scores.append(float(itm_scores.max()))
                
                score_itm = max(all_itm_scores) if all_itm_scores else 0.0
            else:
                score_itm = 0.0

            # Store in cache (with descriptors for A-EQA)
            score_cache[f.frontier_id] = {
                "score_itm": score_itm,
                "vlm_likelihood": vlm_likelihood_score,
                "descriptors": descriptors  # A-EQA: descriptor 저장
            }
        else:
            logging.info(f"Cache HIT for Frontier {f.frontier_id}.")
            cached_scores = score_cache[f.frontier_id]
            score_itm = cached_scores.get("score_itm", 0.0)
            vlm_likelihood_score = cached_scores["vlm_likelihood"]
            descriptors = cached_scores.get("descriptors", [])

        # Calculate Exploration Score
        unexplored_volume = tsdf_planner.get_unexplored_volume_from_frontier(f)
        normalized_volume_score = np.tanh(unexplored_volume / 1000.0)
        distance = np.linalg.norm(cur_voxel[:2] - f.position) * tsdf_planner._voxel_size
        exploration_score = 0.25 * (normalized_volume_score + (1.0 / (1.0 + distance)))

        # Calculate Semantic Score
        semantic_score = (cfg.w_itm_score * score_itm) + (cfg.w_vlm_score * vlm_likelihood_score)

        scored_frontiers.append({
            "frontier": f,
            "semantic_score": semantic_score,
            "score_itm": score_itm,
            "vlm_likelihood": vlm_likelihood_score,
            "exploration_score": exploration_score,
            "descriptors": descriptors  # A-EQA: descriptor 포함
        })

    if not scored_frontiers:
        return None, None, []
        
    # Select the best frontier based on the logic
    scored_frontiers.sort(key=lambda x: x["semantic_score"], reverse=True)
    
    best_frontier_data = scored_frontiers[0]
    
    # Condition 1: All scores are low, resort to pure exploration
    if best_frontier_data["semantic_score"] < cfg.low_score_threshold:
        logging.info(f"All frontier scores are below threshold {cfg.low_score_threshold}. "
                     f"Switching to exploration mode.")
        scored_frontiers.sort(key=lambda x: x["exploration_score"], reverse=True)
        best_frontier_data = scored_frontiers[0]
        logging.info(f"Best Frontier (by Exploration): {best_frontier_data['frontier'].frontier_id} "
                     f"with Exploration Score: {best_frontier_data['exploration_score']:.3f}")

    # Condition 2: Tie-breaking for top two candidates
    else:
        top_score = scored_frontiers[0]["semantic_score"]
        tie_candidates = [
            f for f in scored_frontiers 
            if top_score - f["semantic_score"] < cfg.tie_breaking_threshold
        ]
        
        if len(tie_candidates) > 1:
            logging.info(f"{len(tie_candidates)} frontiers are in a tie-breaking contention. "
                         f"Using exploration score to resolve.")
            
            best_in_tie = max(
                tie_candidates, 
                key=lambda x: x["semantic_score"] + x["exploration_score"]
            )
            
            if best_in_tie['frontier'].frontier_id != best_frontier_data['frontier'].frontier_id:
                original_winner = best_frontier_data
                logging.info(f"Tie-breaker chose Frontier {best_in_tie['frontier'].frontier_id} "
                             f"(Combined: {best_in_tie['semantic_score'] + best_in_tie['exploration_score']:.3f}) "
                             f"over Frontier {original_winner['frontier'].frontier_id} "
                             f"(Combined: {original_winner['semantic_score'] + original_winner['exploration_score']:.3f}).")
            
            best_frontier_data = best_in_tie
    
    chosen_frontier = best_frontier_data["frontier"]
    
    logging.info(f"Best Frontier: {chosen_frontier.frontier_id} with Semantic Score: {best_frontier_data['semantic_score']:.3f} "
                 f"(ITM: {best_frontier_data['score_itm']:.3f}, VLM: {best_frontier_data['vlm_likelihood']:.3f}, "
                 f"Exploration: {best_frontier_data['exploration_score']:.3f})")

    # Prepare visualization log
    candidates_info = None
    if cfg.save_visualization:
        candidates_for_log = []
        scores_for_log = {}
        
        frontier_to_rgb = {d['frontier'].frontier_id: d['rgb'] for d in frontiers_with_data}

        for f_data in scored_frontiers:
            f = f_data["frontier"]
            candidate_name = f"frontier_{f.frontier_id}"
            
            if f.frontier_id in frontier_to_rgb:
                candidates_for_log.append({
                    "rgb": frontier_to_rgb[f.frontier_id],
                    "name": candidate_name
                })
                scores_for_log[candidate_name] = f_data['semantic_score']

        if candidates_for_log:
            chosen_name_for_log = f"frontier_{chosen_frontier.frontier_id}"
            bfd = best_frontier_data
            caption = (
                f"Goal: {original_goal_str}\n"
                f"Chosen: {chosen_name_for_log} | Score: {bfd['semantic_score']:.2f}\n"
                f"ITM: {bfd['score_itm']:.2f}, VLM: {bfd['vlm_likelihood']:.2f}, Exploration: {bfd['exploration_score']:.2f}"
            )
            
            candidates_info = {
                "save_dir": selection_dir,
                "candidates": candidates_for_log,
                "chosen_name": chosen_name_for_log,
                "scores": scores_for_log,
                "caption": caption,
                "filename_prefix": ""
            }

    # A-EQA: Query-aware selective descriptor accumulation
    # Instead of just using the best frontier's descriptors, we collect descriptors
    # from all frontiers that are semantically relevant to the question.
    # This maximizes information gain while keeping memory compact.
    accumulated_descriptors = []
    
    # Strategy: Collect descriptors from frontiers with semantic score above threshold
    # This ensures we gather diverse, question-relevant information from the environment
    relevance_threshold = cfg.get("descriptor_relevance_threshold", 0.2)
    
    for f_data in scored_frontiers:
        if f_data["semantic_score"] >= relevance_threshold:
            descriptors = f_data.get("descriptors", [])
            if descriptors:
                accumulated_descriptors.extend(descriptors)
                logging.info(f"Frontier {f_data['frontier'].frontier_id}: "
                           f"Adding {len(descriptors)} descriptors (score: {f_data['semantic_score']:.3f})")
    
    if not accumulated_descriptors:
        # Fallback: if no frontier meets threshold, use the best one
        accumulated_descriptors = best_frontier_data.get("descriptors", [])
        logging.info(f"No frontier above threshold {relevance_threshold}. "
                   f"Using best frontier's {len(accumulated_descriptors)} descriptors.")
    else:
        logging.info(f"Collected {len(accumulated_descriptors)} descriptors from "
                   f"{sum(1 for f in scored_frontiers if f['semantic_score'] >= relevance_threshold)} relevant frontiers.")
    
    return chosen_frontier, candidates_info, accumulated_descriptors


def _check_answer_correctness(predicted_answer: str, ground_truth: str) -> bool:
    """
    A-EQA 답변 정확도를 평가합니다.
    
    Args:
        predicted_answer: Agent가 생성한 답변
        ground_truth: 실제 정답
    
    Returns:
        True if correct, False otherwise
    """
    if not predicted_answer or not ground_truth:
        return False
    
    # Normalize strings: lowercase, strip whitespace
    pred = predicted_answer.lower().strip()
    gt = ground_truth.lower().strip()
    
    # Exact match
    if pred == gt:
        return True
    
    # Partial match: check if GT is in prediction or vice versa
    # (A-EQA answers are often short, e.g., "red", "blue", "yes", "no")
    if gt in pred or pred in gt:
        return True
    
    # TODO: Could add more sophisticated matching:
    # - Fuzzy string matching (Levenshtein distance)
    # - Semantic similarity (e.g., "couch" == "sofa")
    # - Number matching (e.g., "2" == "two")
    
    return False


def _generate_answer_from_descriptors(
    app_state: AppState,
    question: str,
    accumulated_descriptors: set,
) -> Tuple[str, float]:
    """
    누적된 descriptor들을 기반으로 질문에 대한 답변을 생성합니다.
    
    Args:
        accumulated_descriptors: Set of unique descriptors (already deduplicated)
    
    Returns:
        (answer, confidence_score)
    """
    try:
        # Convert set to list for processing
        unique_descriptors = list(accumulated_descriptors)
        
        # 질문과 가장 관련성 높은 descriptor 선별 (상위 20개)
        if len(unique_descriptors) > 20:
            # ITM score로 관련성 계산
            relevance_scores = []
            for desc in unique_descriptors:
                score = app_state.itm.text_text_scores(question, [desc])
                if score is not None and score.size > 0:
                    relevance_scores.append((desc, float(score.max())))
                else:
                    relevance_scores.append((desc, 0.0))
            
            # 상위 20개 선택
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            selected_descriptors = [desc for desc, _ in relevance_scores[:20]]
        else:
            selected_descriptors = unique_descriptors
        
        # GPT로 답변 생성
        descriptors_text = "\n".join([f"- {desc}" for desc in selected_descriptors])
        prompt = (
            f"You are a helpful assistant answering questions about an indoor environment.\n\n"
            f"Question: {question}\n\n"
            f"Scene descriptions extracted from explored areas:\n{descriptors_text}\n\n"
            f"Based on these scene descriptions, answer the question concisely.\n"
            f"Return a JSON object with:\n"
            f"- 'answer': your answer (string, concise)\n"
            f"- 'confidence': confidence score 0.0-1.0 (float)\n"
            f"If you cannot answer confidently, set confidence to 0.3 or lower."
        )
        
        response = app_state.desc_extractor.client.chat.completions.create(
            model=app_state.desc_extractor.cfg.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        
        response_str = response.choices[0].message.content
        if not response_str:
            return "", 0.0
        
        response_json = json.loads(response_str)
        answer = response_json.get("answer", "")
        confidence = float(response_json.get("confidence", 0.0))
        
        return answer, confidence
    
    except Exception as e:
        logging.error(f"Error generating answer from descriptors: {e}", exc_info=True)
        return "", 0.0


def get_relevant_observations(
    frontier: Frontier,
    observations: List[Dict[str, Any]],
    angle_threshold_deg: float,
) -> List[Dict[str, Any]]:
    """
    주어진 관측 리스트 내에서 특정 프론티어와 관련된 모든 관측들을 찾습니다.
    """
    relevant_obs = []
    frontier_angle = np.arctan2(frontier.orientation[1], frontier.orientation[0])
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    for obs_data in observations:
        obs_angle = obs_data.get("angle")
        if obs_angle is None:
            continue
        
        # 각도 차이 계산 (0~π 범위로 정규화)
        angle_diff = abs(frontier_angle - obs_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        # 각도 차이가 임계값 이내이면 관련 관측으로 추가
        if angle_diff <= angle_threshold_rad:
            relevant_obs.append(obs_data)

    logging.debug(f"Frontier {frontier.frontier_id}: Found {len(relevant_obs)} relevant observations. "
                 f"Frontier angle: {frontier_angle:.1f}°")
    return relevant_obs


def main(cfg_file: str, start_ratio: float, end_ratio: float, question_id: Optional[str] = None):
    """메인 실행 함수: 준비 -> 평가"""
    app_state = setup(cfg_file, start_ratio, end_ratio)
    run_evaluation(app_state, start_ratio, end_ratio, question_id)
    logging.info("All questions finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/eval_aeqa_ours.yaml", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--question_id", help="Run evaluation for a specific question ID", default=None, type=str)
    args = parser.parse_args()
    
    # If question_id is an empty string or the unsubstituted variable from launch.json, treat it as None
    if args.question_id == "" or args.question_id == "${input:questionId}":
        args.question_id = None

    logging.info(f"***** Running No-Memory MZSON methodology on AEQA *****")
    main(args.cfg_file, args.start_ratio, args.end_ratio, args.question_id)
