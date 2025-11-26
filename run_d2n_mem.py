import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# OpenAI API 키 설정
from src.const import OPENAI_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_KEY

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import habitat_sim
import numpy as np
import torch
from omegaconf import OmegaConf
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf, pos_habitat_to_normal
from src.geom import get_cam_intr, get_scene_bnds
from src import TSDFPlannerMem as TSDFPlanner
from src.tsdf_planner import SnapShot
from src.scene_goatbench import Scene
from src.utils import (
    calc_agent_subtask_distance,
    get_pts_angle_goatbench,
)
from src.goatbench_utils import prepare_goatbench_navigation_goals
from src.logger_mzson import MZSONLogger
from src.siglip_itm import SigLipITM
from src.descriptor_extractor import DescriptorExtractor, DescriptorExtractorConfig
from src.data_models import Frontier


class ElapsedTimeFormatter(logging.Formatter):
    """로그 시간(UTC)을 시작 시점으로부터의 경과 시간으로 표시합니다."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        elapsed_seconds = record.created - self.start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


@dataclass
class AnalyzedObservation:
    """메모리에 저장되는 관측 기반 스냅샷 정보."""

    name: str
    map_pos: Tuple[int, int]
    angle: float
    descriptors: List[str]
    rgb: np.ndarray = field(repr=False)
    key_objects: List[str] = field(default_factory=list)
    best_itm_score: float = 0.0
    vlm_likelihood: float = 0.0
    timestamp: float = field(default_factory=time.time)
    object_infos: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AppState:
    """실험 전역 상태(모델, 설정, 메모리 등)를 관리합니다."""

    cfg: OmegaConf
    device: torch.device
    logger: MZSONLogger
    itm: SigLipITM
    desc_extractor: DescriptorExtractor
    detection_model: YOLOWorld
    sam_predictor: SAM
    cam_intr: np.ndarray
    min_depth: Optional[float]
    max_depth: Optional[float]
    observation_memory: Dict[str, AnalyzedObservation] = field(default_factory=dict)


def setup(cfg_file: str, start_ratio: float, end_ratio: float, split: int) -> AppState:
    """설정 로딩, 로깅 초기화, 모델 로딩 등을 수행합니다."""
    cfg = OmegaConf.load(cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if os.path.abspath(cfg_file) != os.path.abspath(
        os.path.join(cfg.output_dir, os.path.basename(cfg_file))
    ):
        os.system(f"cp {cfg_file} {cfg.output_dir}")

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{start_ratio:.2f}_{end_ratio:.2f}_{split}.log"
    )
    main_file_handler = logging.FileHandler(logging_path, mode="w")
    main_file_handler.setFormatter(formatter)
    root_logger.addHandler(main_file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    logger = MZSONLogger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )

    cam_intr = get_cam_intr(cfg.hfov, cfg.img_height, cfg.img_width)
    min_depth = cfg.min_depth if hasattr(cfg, "min_depth") else None
    max_depth = cfg.max_depth if hasattr(cfg, "max_depth") else None

    logging.info("Setup complete.")

    return AppState(
        cfg=cfg,
        device=device,
        logger=logger,
        itm=itm,
        desc_extractor=desc_extractor,
        detection_model=detection_model,
        sam_predictor=sam_predictor,
        cam_intr=cam_intr,
        min_depth=min_depth,
        max_depth=max_depth,
        observation_memory={},
    )


def run_evaluation(
    app_state: AppState,
    start_ratio: float,
    end_ratio: float,
    split: int,
    scene_id: Optional[str] = None,
):
    """데이터셋을 순회하며 평가를 실행합니다."""
    cfg = app_state.cfg
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    scene_data_list = os.listdir(cfg.test_data_dir)

    if scene_id:
        target_scene_file = None
        for f in scene_data_list:
            scene_name_from_file = f.split(".")[0]
            if scene_name_from_file in scene_id:
                target_scene_file = f
                break

        if target_scene_file:
            scene_data_list = [target_scene_file]
            logging.info(f"Running for single scene: {scene_id}")
        else:
            logging.error(
                f"Scene ID {scene_id} not found in {cfg.test_data_dir}. Exiting."
            )
            return
    else:
        num_scene = len(scene_data_list)
        random.shuffle(scene_data_list)
        scene_data_list = scene_data_list[
            int(start_ratio * num_scene) : int(end_ratio * num_scene)
        ]

    num_episode = sum(
        len(json.load(open(os.path.join(cfg.test_data_dir, f), "r"))["episodes"])
        for f in scene_data_list
    )
    logging.info(
        f"Total episodes: {num_episode}; Selected scenes: {len(scene_data_list)}"
    )

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")
    root_logger = logging.getLogger()
    main_file_handler = None
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and "log_" in os.path.basename(
            handler.baseFilename
        ):
            main_file_handler = handler
            break

    for scene_data_file in scene_data_list:
        scene_name = scene_data_file.split(".")[0]
        target_scene_id_candidates = [sid for sid in all_scene_ids if scene_name in sid]
        if not target_scene_id_candidates:
            logging.warning(f"Scene {scene_name} not found in scene data path. Skip.")
            continue
        scene_identifier = target_scene_id_candidates[0]

        scene_file_handler = None
        try:
            if main_file_handler:
                root_logger.removeHandler(main_file_handler)

            log_dir = os.path.join(app_state.cfg.output_dir, scene_identifier)
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{scene_identifier}.log")
            scene_file_handler = logging.FileHandler(log_path, mode="w")
            scene_file_handler.setFormatter(formatter)
            root_logger.addHandler(scene_file_handler)

            scene_data = json.load(
                open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
            )
            scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
            all_navigation_goals = scene_data["goals"]

            scene = None
            for episode_idx, episode in enumerate(scene_data["episodes"]):
                episode_id = episode["episode_id"]
                logging.info(
                    f"Starting Episode {episode_idx + 1}/{len(scene_data['episodes'])} "
                    f"in scene {scene_identifier}"
                )

                if scene is not None:
                    scene.close()
                    del scene

                all_subtask_goal_types, all_subtask_goals = (
                    prepare_goatbench_navigation_goals(
                        scene_name=scene_name,
                        episode=episode,
                        all_navigation_goals=all_navigation_goals,
                    )
                )

                finished_subtask_ids = list(app_state.logger.success_by_snapshot.keys())
                finished_episode_subtask = [
                    sid
                    for sid in finished_subtask_ids
                    if sid.startswith(f"{scene_identifier}_{episode_id}_")
                ]
                if len(finished_episode_subtask) >= len(all_subtask_goals):
                    logging.info(
                        f"Scene {scene_identifier} Episode {episode_id} already done!"
                    )
                    continue

                pts, angle = get_pts_angle_goatbench(
                    episode["start_position"], episode["start_rotation"]
                )

                scene = Scene(
                    scene_identifier,
                    cfg,
                    cfg_cg,
                    app_state.detection_model,
                    app_state.sam_predictor,
                    vlm_model=app_state.itm,
                    device=app_state.device,
                )

                floor_height = pts[1]
                tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
                max_steps = max(
                    int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio), 50
                )
                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height=floor_height,
                    floor_height_offset=0,
                    pts_init=pts,
                    init_clearance=cfg.init_clearance * 2,
                    save_visualization=cfg.save_visualization,
                )

                episode_context = {
                    "start_position": pts,
                    "floor_height": floor_height,
                    "tsdf_bounds": tsdf_bnds,
                    "visited_positions": [],
                    "observations_history": [],
                    "step_count": 0,
                    "subtask_observation_cache": {},
                    "subtask_id": f"{scene_identifier}_{episode_id}",
                    "subtask_goal_str": "N/A",
                    "target_objects": [],
                }

                app_state.logger.init_episode(
                    scene_id=scene_identifier, episode_id=episode_id
                )

                episode_context["tsdf_planner"] = tsdf_planner

                global_step = 0
                for subtask_idx, (goal_type, subtask_goal) in enumerate(
                    zip(all_subtask_goal_types, all_subtask_goals)
                ):
                    subtask_id = f"{scene_identifier}_{episode_id}_{subtask_idx}"

                    if subtask_id in app_state.logger.success_by_snapshot:
                        logging.info(
                            f"Subtask {subtask_idx + 1}/{len(all_subtask_goals)} already done!"
                        )
                        continue

                    logging.info(
                        f"Reusing TSDF planner for subtask {subtask_idx + 1}; resetting navigation targets."
                    )
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None
                    episode_context["tsdf_planner"] = tsdf_planner

                    subtask_metadata = app_state.logger.init_subtask(
                        subtask_id=subtask_id,
                        goal_type=goal_type,
                        subtask_goal=subtask_goal,
                        pts=pts,
                        scene=scene,
                        tsdf_planner=tsdf_planner,
                    )

                    question = subtask_metadata.get("question", "N/A")
                    logging.info(
                        f"\nSubtask {subtask_idx + 1}/{len(all_subtask_goals)}: {question}"
                    )

                    task_result = run_subtask(
                        app_state=app_state,
                        subtask_id=subtask_id,
                        subtask_metadata=subtask_metadata,
                        scene=scene,
                        episode_context=episode_context,
                        pts=pts,
                        angle=angle,
                        max_steps=max_steps,
                        global_step=global_step,
                        tsdf_planner=tsdf_planner,
                    )
                    global_step = task_result.get("final_global_step", global_step)
                    pts = task_result.get("final_position", pts)
                    angle = task_result.get("final_angle", angle)

                app_state.logger.save_results()
                if not cfg.save_visualization:
                    os.system(f"rm -r {app_state.logger.get_episode_dir()}")

            if scene is not None:
                scene.close()
                del scene

            logging.info(f"Scene {scene_identifier} finished. Clearing observation memory.")
            app_state.observation_memory.clear()

        finally:
            if scene_file_handler:
                scene_file_handler.close()
                root_logger.removeHandler(scene_file_handler)
            if main_file_handler:
                root_logger.addHandler(main_file_handler)

    app_state.logger.save_results()
    app_state.logger.aggregate_results()


def _collect_memory_candidates(
    app_state: AppState,
    episode_context: Dict[str, Any],
    primary_target: Optional[str],
    secondary_targets: List[str],
) -> List[Dict[str, Any]]:
    used_names: Set[str] = episode_context.setdefault("memory_used_names", set())
    allowed_names: Optional[Set[str]] = episode_context.get("available_memory_names")

    primary_targets = (
        [primary_target.strip()]
        if isinstance(primary_target, str) and primary_target.strip()
        else []
    )
    secondary_targets_clean = [
        str(t).strip()
        for t in secondary_targets
        if isinstance(t, str) and str(t).strip()
    ]

    memory_candidates: List[Dict[str, Any]] = []

    for mem in app_state.observation_memory.values():
        if allowed_names is not None and mem.name not in allowed_names:
            continue
        if mem.name in used_names:
            continue

        descriptors = mem.descriptors or []
        combined_score, primary_score, secondary_score = app_state.desc_extractor.compute_itm_scores(
            primary_targets,
            secondary_targets_clean,
            descriptors,
        )

        map_pos = np.asarray(mem.map_pos, dtype=int)
        object_voxels: List[np.ndarray] = []
        for info in mem.object_infos:
            center_voxel = info.get("center_voxel")
            if center_voxel is not None:
                object_voxels.append(np.asarray(center_voxel, dtype=int))

        memory_candidates.append(
            {
                "observation": mem,
                "map_pos": map_pos,
                "object_voxels": object_voxels,
                "combined_score": combined_score,
                "primary_score": primary_score,
                "secondary_score": secondary_score,
            }
        )

    memory_candidates.sort(key=lambda c: c["combined_score"], reverse=True)
    return memory_candidates


def run_subtask(
    app_state: AppState,
    subtask_id: str,
    subtask_metadata: Dict[str, Any],
    scene: Scene,
    episode_context: Dict[str, Any],
    pts: np.ndarray,
    angle: float,
    max_steps: int,
    global_step: int,
    tsdf_planner: TSDFPlanner,
) -> Dict[str, Any]:
    """단일 서브태스크를 수행합니다."""
    episode_context["subtask_full_observation_history"] = {}
    episode_context["frontier_score_cache"] = {}
    episode_context["available_memory_names"] = set(app_state.observation_memory.keys())
    episode_context["memory_used_names"] = set()

    goal = subtask_metadata.get("goal")
    goal_type = subtask_metadata.get("goal_type")
    original_goal_str = subtask_metadata.get("question", subtask_id)

    current_pts = pts.copy()
    current_angle = angle
    success_by_distance = False

    target_objects: List[str] = []
    secondary_targets: List[str] = []
    primary_target: Optional[str] = None
    if goal_type == "object":
        raw_targets = goal if isinstance(goal, list) else [goal]
        if raw_targets:
            primary_target = next(
                (
                    str(t).strip()
                    for t in raw_targets
                    if isinstance(t, str) and str(t).strip()
                ),
                None,
            )
    elif goal_type == "description":
        logging.info("Goal is a description. Extracting target objects...")
        main_targets, context_objects = (
            app_state.desc_extractor.extract_target_objects_from_description(goal)
        )
        if main_targets:
            primary_target = next(
                (
                    str(t).strip()
                    for t in main_targets
                    if isinstance(t, str) and str(t).strip()
                ),
                None,
            )
        if context_objects:
            secondary_candidate = next(
                (
                    str(c).strip()
                    for c in context_objects
                    if isinstance(c, str) and str(c).strip()
                ),
                None,
            )
            if secondary_candidate:
                secondary_targets.append(secondary_candidate)
        if primary_target is None and isinstance(goal, list) and goal:
            candidate = goal[0]
            if isinstance(candidate, str) and candidate.strip():
                primary_target = candidate.strip()
        if primary_target is None and isinstance(goal, str) and goal.strip():
            primary_target = goal.strip()
    elif goal_type == "image":
        logging.info("Goal is an image. Extracting keywords for ITM scoring...")
        keywords = app_state.desc_extractor.extract_keywords_from_image_modified(goal)
        logging.info(f"Extracted keywords from image goal: {keywords}")
        if keywords:
            ordered_keywords = [
                kw.strip()
                for kw in keywords
                if isinstance(kw, str)
                and kw.strip()
                and kw.lower().strip() != "detection failed"
            ]
            if ordered_keywords:
                primary_target = ordered_keywords[0]
                if len(ordered_keywords) > 1:
                    secondary_targets.extend(ordered_keywords[1:2])

    target_objects = []
    if primary_target:
        target_objects.append(primary_target)
    for sec in secondary_targets:
        if sec and sec not in target_objects:
            target_objects.append(sec)

    if not target_objects:
        logging.warning(
            "No valid primary target could be determined; proceeding without semantic target."
        )

    logging.info(f"Refined Target Objects: {target_objects}")
    if secondary_targets:
        logging.info(f"Secondary Targets: {secondary_targets}")
    episode_context["target_objects"] = (
        target_objects if isinstance(target_objects, list) else []
    )
    episode_context["secondary_targets"] = secondary_targets
    episode_context["primary_target"] = primary_target
    episode_context["subtask_goal_str"] = original_goal_str

    memory_candidates = _collect_memory_candidates(
        app_state=app_state,
        episode_context=episode_context,
        primary_target=primary_target,
        secondary_targets=secondary_targets,
    )
    memory_threshold = float(getattr(app_state.cfg, "memory_navigation_threshold", 0.7))
    used_memory_names: Set[str] = episode_context.setdefault("memory_used_names", set())

    active_memory_record: Optional[Dict[str, Any]] = None
    if memory_candidates and memory_candidates[0]["combined_score"] >= memory_threshold:
        best_mem = memory_candidates.pop(0)
        object_voxel = (
            best_mem["object_voxels"][0] if best_mem["object_voxels"] else None
        )
        success = tsdf_planner.set_target_point_from_observation(
            best_mem["map_pos"],
            current_pts=current_pts,
            object_voxel=object_voxel,
            pathfinder=scene.pathfinder,
        )
        if success:
            used_memory_names.add(best_mem["observation"].name)
            active_memory_record = {
                "name": best_mem["observation"].name,
                "map_pos": best_mem["map_pos"].tolist(),
                "score": best_mem["combined_score"],
            }
            logging.info(
                "Warm-starting from memory candidate '%s' (score %.3f).",
                best_mem["observation"].name,
                best_mem["combined_score"],
            )
        else:
            logging.warning(
                "Failed to set navigation target from memory candidate '%s'.",
                best_mem["observation"].name,
            )

    episode_context["active_memory_target"] = active_memory_record

    cnt_step = 0
    while cnt_step < max_steps:
        cnt_step += 1
        global_step += 1

        candidates_info = None

        if tsdf_planner.max_point is None:
            logging.info("No long-term target. Stop, Scan, and Select.")

            current_step_observations = _observe_and_update_maps(
                scene=scene,
                tsdf_planner=tsdf_planner,
                current_pts=current_pts,
                current_angle=current_angle,
                cnt_step=cnt_step,
                cfg=app_state.cfg,
                cam_intr=app_state.cam_intr,
                min_depth=app_state.min_depth,
                max_depth=app_state.max_depth,
                eps_frontier_dir=app_state.logger.get_frontier_dir(),
                app_state=app_state,
                episode_context=episode_context,
            )

            for obs in current_step_observations:
                episode_context["subtask_full_observation_history"][obs["name"]] = obs

            for frontier in tsdf_planner.frontiers:
                if frontier.source_observation_name is None:
                    relevant_obs_list = get_relevant_observations(
                        frontier,
                        current_step_observations,
                        app_state.cfg.relevant_view_angle_threshold_deg,
                    )
                    if relevant_obs_list:
                        selected_obs = relevant_obs_list[0]
                        frontier.source_observation_name = selected_obs["name"]
                        frontier.cam_pose = selected_obs["cam_pose"]
                        frontier.depth = selected_obs["depth"]

            active_frontier_ids = {f.frontier_id for f in tsdf_planner.frontiers}
            required_obs_names = {
                f.source_observation_name
                for f in tsdf_planner.frontiers
                if f.source_observation_name
            }
            episode_context["subtask_observation_cache"] = {
                name: obs
                for name, obs in episode_context[
                    "subtask_full_observation_history"
                ].items()
                if name in required_obs_names
            }
            current_cache = episode_context.get("frontier_score_cache", {})
            episode_context["frontier_score_cache"] = {
                fid: score_data
                for fid, score_data in current_cache.items()
                if fid in active_frontier_ids
            }

            if not required_obs_names and tsdf_planner.frontiers:
                if episode_context["subtask_full_observation_history"]:
                    any_obs = next(
                        iter(episode_context["subtask_full_observation_history"].values())
                    )
                    for frontier in tsdf_planner.frontiers:
                        if frontier.source_observation_name is None:
                            frontier.source_observation_name = any_obs["name"]
                            frontier.cam_pose = any_obs["cam_pose"]
                            frontier.depth = any_obs["depth"]
                    episode_context["subtask_observation_cache"][any_obs["name"]] = (
                        any_obs
                    )

            selection_dir = os.path.join(app_state.logger.subtask_dir, "selection")
            memory_candidates_for_selection = _collect_memory_candidates(
                app_state=app_state,
                episode_context=episode_context,
                primary_target=primary_target,
                secondary_targets=secondary_targets,
            )
            chosen_candidate, candidates_info = _select_next_target(
                app_state=app_state,
                episode_context=episode_context,
                current_pts=current_pts,
                goal=goal,
                goal_type=goal_type,
                original_goal_str=original_goal_str,
                tsdf_planner=tsdf_planner,
                selection_dir=selection_dir,
                target_objects=target_objects,
                memory_candidates=memory_candidates_for_selection,
            )
            if not chosen_candidate:
                logging.warning(
                    "Failed to select a new target candidate. Will re-scan next step."
                )

            if chosen_candidate:
                if chosen_candidate["type"] == "frontier":
                    set_ok = tsdf_planner.set_next_navigation_point(
                        choice=chosen_candidate["frontier"],
                        pts=current_pts,
                        objects=scene.objects,
                        cfg=app_state.cfg.planner,
                        pathfinder=scene.pathfinder,
                    )
                else:
                    mem_choice = chosen_candidate["memory"]
                    object_voxel = (
                        mem_choice["object_voxels"][0]
                        if mem_choice["object_voxels"]
                        else None
                    )
                    set_ok = tsdf_planner.set_target_point_from_observation(
                        mem_choice["map_pos"],
                        current_pts=current_pts,
                        object_voxel=object_voxel,
                        pathfinder=scene.pathfinder,
                    )
                    used_memory_names.add(mem_choice["observation"].name)
                    if set_ok:
                        episode_context["active_memory_target"] = {
                            "name": mem_choice["observation"].name,
                            "map_pos": mem_choice["map_pos"].tolist(),
                            "score": mem_choice["combined_score"],
                        }

                if not set_ok:
                    logging.warning(
                        "Failed to set navigation point for newly chosen target. "
                        "Will retry next step."
                    )
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

        elif tsdf_planner.target_point is None:
            logging.info("Committed target exists. Calculating next waypoint.")
            set_ok = tsdf_planner.set_next_navigation_point(
                choice=tsdf_planner.max_point,
                pts=current_pts,
                objects=scene.objects,
                cfg=app_state.cfg.planner,
                pathfinder=scene.pathfinder,
            )
            if not set_ok:
                logging.warning(
                    "Failed to find path to committed target. Clearing for re-evaluation."
                )
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None

        else:
            logging.info(f"Continuing towards current target: {tsdf_planner.target_point}")

        target_info = (
            f"| Target: {tsdf_planner.target_point}"
            if tsdf_planner.target_point is not None
            else ""
        )
        logging.info(
            f"\nStep {cnt_step}/{max_steps}, Global step: {global_step} {target_info}"
        )

        step_vals = tsdf_planner.agent_step(
            pts=current_pts,
            angle=current_angle,
            objects=scene.objects,
            snapshots=scene.snapshots,
            pathfinder=scene.pathfinder,
            cfg=app_state.cfg.planner
            if hasattr(app_state.cfg, "planner")
            else app_state.cfg,
            save_visualization=app_state.cfg.save_visualization,
        )

        if step_vals[0] is None:
            logging.warning(
                "Agent step failed. Clearing targets to force re-evaluation."
            )
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            continue

        current_pts, current_angle, _, fig, _, waypoint_arrived = step_vals
        app_state.logger.log_step(pts_voxel=tsdf_planner.habitat2voxel(current_pts)[:2])
        episode_context["visited_positions"].append(current_pts.copy())

        active_mem_info = episode_context.get("active_memory_target")
        if active_mem_info:
            target_voxel = None
            if (
                isinstance(tsdf_planner.max_point, SnapShot)
                and hasattr(tsdf_planner.max_point, "memory_target")
                and tsdf_planner.max_point.memory_target is not None
            ):
                target_voxel = np.asarray(tsdf_planner.max_point.memory_target)
            else:
                target_voxel = np.asarray(active_mem_info["map_pos"])

            current_voxel = tsdf_planner.habitat2voxel(current_pts)[:2]
            dist_to_memory = (
                np.linalg.norm(current_voxel[:2] - target_voxel[:2])
                * tsdf_planner._voxel_size
            )
            arrival_thresh = float(
                getattr(app_state.cfg, "memory_arrival_distance", app_state.cfg.success_distance)
            )
            if dist_to_memory <= arrival_thresh:
                logging.info(
                    "Arrived near memory target '%s' (distance %.3f m). Switching to exploration.",
                    active_mem_info["name"],
                    dist_to_memory,
                )
                episode_context["active_memory_target"] = None
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None

        if waypoint_arrived:
            logging.info("Intermediate waypoint reached.")
            tsdf_planner.target_point = None

            if tsdf_planner.max_point is not None:
                current_voxel_pos = tsdf_planner.normal2voxel(
                    pos_habitat_to_normal(current_pts)
                )
                dist_to_max_point_m = (
                    np.linalg.norm(current_voxel_pos[:2] - tsdf_planner.max_point.position)
                    * tsdf_planner._voxel_size
                )
                if dist_to_max_point_m < 0.5:
                    logging.info(
                        "Vicinity of long-term target reached. Clearing for full re-evaluation."
                    )
                    tsdf_planner.max_point = None

        agent_subtask_distance = calc_agent_subtask_distance(
            current_pts, subtask_metadata["viewpoints"], scene.pathfinder
        )
        success_by_distance = agent_subtask_distance < app_state.cfg.success_distance

        if app_state.cfg.save_visualization and fig is not None:
            memory_targets_for_vis: Optional[List[Tuple[int, int]]] = None
            active_mem_for_vis = episode_context.get("active_memory_target")
            if active_mem_for_vis and active_mem_for_vis.get("map_pos"):
                mem_pos_arr = np.asarray(active_mem_for_vis["map_pos"]).astype(int)
                memory_targets_for_vis = [(int(mem_pos_arr[0]), int(mem_pos_arr[1]))]
            elif (
                isinstance(tsdf_planner.max_point, SnapShot)
                and hasattr(tsdf_planner.max_point, "memory_target")
                and tsdf_planner.max_point.memory_target is not None
            ):
                mem_pos_arr = np.asarray(tsdf_planner.max_point.memory_target).astype(int)
                memory_targets_for_vis = [(int(mem_pos_arr[0]), int(mem_pos_arr[1]))]

            if success_by_distance:
                logging.info(
                    f"SUCCESS: Distance condition met at step {global_step} "
                    f"({agent_subtask_distance:.2f}m < {app_state.cfg.success_distance:.1f}m). "
                    "Stopping."
                )
                success_voxel_pos = tsdf_planner.habitat2voxel(current_pts)[:2]
                tsdf_planner.target_point = success_voxel_pos

                tsdf_planner.max_point = SnapShot(
                    image="success_position",
                    color=(0, 255, 0),
                    obs_point=success_voxel_pos,
                    position=success_voxel_pos,
                )

                success_step_vals = tsdf_planner.agent_step(
                    pts=current_pts,
                    angle=current_angle,
                    objects=scene.objects,
                    snapshots=scene.snapshots,
                    pathfinder=scene.pathfinder,
                    cfg=app_state.cfg.planner
                    if hasattr(app_state.cfg, "planner")
                    else app_state.cfg,
                    save_visualization=True,
                )

                if success_step_vals[0] is not None:
                    success_fig = success_step_vals[3]
                    if success_fig is not None:
                        app_state.logger.save_success_visualization(
                            subtask_id=subtask_metadata["subtask_id"],
                            subtask_metadata=subtask_metadata,
                            fig=success_fig,
                        )

                tsdf_planner.target_point = None
                tsdf_planner.max_point = None
            else:
                app_state.logger.log_step_visualizations(
                    global_step=global_step,
                    subtask_id=subtask_metadata["subtask_id"],
                    subtask_metadata=subtask_metadata,
                    fig=fig,
                    candidates_info=candidates_info,
                    memory_targets=memory_targets_for_vis,
                )

        if success_by_distance:
            break

        if cnt_step >= max_steps:
            logging.warning(f"Max steps reached ({max_steps}). Ending subtask.")
            break

    success_by_snapshot = False
    if success_by_distance:
        logging.info(f"Task completed successfully in {cnt_step} steps!")
    else:
        logging.info(f"Task failed after {cnt_step} steps")

    app_state.logger.log_subtask_result(
        success_by_snapshot=success_by_snapshot,
        success_by_distance=success_by_distance,
        subtask_id=subtask_id,
        gt_subtask_explore_dist=subtask_metadata.get("gt_subtask_explore_dist", 0),
        goal_type=subtask_metadata.get("goal_type", "unknown"),
        n_filtered_snapshots=0,
        n_total_snapshots=0,
        n_total_frames=cnt_step,
    )

    return {
        "success": success_by_distance,
        "final_position": current_pts,
        "final_angle": current_angle,
        "steps_taken": cnt_step,
        "final_global_step": global_step,
    }


def _observe_and_update_maps(
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    current_pts: np.ndarray,
    current_angle: float,
    cnt_step: int,
    cfg: OmegaConf,
    cam_intr: np.ndarray,
    min_depth: Optional[float],
    max_depth: Optional[float],
    eps_frontier_dir: str,
    app_state: AppState,
    episode_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    360도 스캔을 수행하고 TSDF / 프론티어 맵을 업데이트하며,
    장기 메모리를 최신 스냅샷과 동기화합니다.
    """
    num_scan_views = cfg.num_scan_views
    angle_increment = np.deg2rad(360 / max(num_scan_views, 1))
    all_angles = [current_angle + (i * angle_increment) for i in range(num_scan_views)]

    current_map_pos = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))
    current_step_observations: List[Dict[str, Any]] = []
    added_object_ids: Set[int] = set()

    frame_counter_key = "scene_frame_counter"
    frame_counter = episode_context.setdefault(frame_counter_key, 0)

    for view_idx, ang in enumerate(all_angles):
        obs, cam_pose = scene.get_observation(current_pts, angle=ang)
        rgb = obs["color_sensor"]
        depth_unfiltered = obs["depth_sensor"]

        obs_file_name = f"{cnt_step}-{view_idx}.png"
        obs_data = {
            "rgb": rgb,
            "angle": ang,
            "cam_pose": cam_pose,
            "name": obs_file_name,
            "depth": depth_unfiltered,
            "map_pos": (int(current_map_pos[0]), int(current_map_pos[1])),
        }
        current_step_observations.append(obs_data)
        scene.all_observations[obs_file_name] = obs_data

        frame_idx = frame_counter + view_idx

        detected_obj_ids: List[int] = []
        try:
            _, added_ids, _ = scene.update_scene_graph(
                image_rgb=rgb,
                depth=depth_unfiltered,
                intrinsics=cam_intr,
                cam_pos=cam_pose,
                pts=current_pts,
                pts_voxel=current_map_pos,
                img_path=obs_file_name,
                frame_idx=frame_idx,
                semantic_obs=None,
                gt_target_obj_ids=None,
            )
            detected_obj_ids = added_ids or []
        except Exception as exc:
            logging.warning(
                f"scene.update_scene_graph failed for {obs_file_name}: {exc}"
            )
        added_object_ids.update(detected_obj_ids)

        try:
            nearby_obj_ids: Set[int] = set()
            include_dist = float(
                getattr(
                    cfg,
                    "snapshot_include_dist",
                    scene.cfg.scene_graph.obj_include_dist + 0.5,
                )
            )
            for obj_id, obj_entry in scene.objects.items():
                if not obj_entry:
                    continue
                bbox = obj_entry.get("bbox")
                if bbox is None:
                    continue
                dist = np.linalg.norm(bbox.center[[0, 2]] - current_pts[[0, 2]])
                if dist <= include_dist:
                    nearby_obj_ids.add(obj_id)
                recent_obs_idx = obj_entry.get("recent_observation_idx")
                if recent_obs_idx is not None:
                    try:
                        recent_obs_name = f"{recent_obs_idx}"
                        if recent_obs_name in scene.frames:
                            recent_obs_pt = scene.frames[recent_obs_name].obs_point
                            dist_recent = np.linalg.norm(
                                recent_obs_pt[[0, 1]] - current_map_pos[:2]
                            )
                            if dist_recent <= include_dist:
                                nearby_obj_ids.add(obj_id)
                    except Exception:
                        pass
            added_object_ids.update(nearby_obj_ids)
        except Exception as exc:
            logging.debug(f"Failed to gather nearby object ids: {exc}")

        try:
            scene.periodic_cleanup_objects(
                frame_idx=frame_idx,
                pts=current_pts,
                goal_obj_ids_mapping=None,
            )
        except Exception as exc:
            logging.debug(f"scene.periodic_cleanup_objects skipped: {exc}")

        frame_snapshot = scene.frames.get(obs_file_name)
        frame_key_objects: List[str] = []
        if frame_snapshot:
            for obj_id in frame_snapshot.full_obj_list.keys():
                obj_meta = scene.objects.get(obj_id)
                if not obj_meta:
                    continue
                class_name = obj_meta.get("class_name")
                if class_name:
                    frame_key_objects.append(class_name)
        frame_key_objects = list(dict.fromkeys(frame_key_objects))

        obs_data["descriptors"] = []
        obs_data["key_objects"] = frame_key_objects
        obs_data["vlm_likelihood"] = 0.0

        if min_depth is not None and max_depth is not None:
            depth = np.where(
                (depth_unfiltered >= min_depth) & (depth_unfiltered <= max_depth),
                depth_unfiltered,
                np.nan,
            )
        else:
            depth = depth_unfiltered

        tsdf_planner.integrate(
            color_im=rgb,
            depth_im=depth,
            cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose),
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )

    episode_context[frame_counter_key] = frame_counter + len(all_angles)

    try:
        if added_object_ids:
            candidate_ids = set(added_object_ids)
            candidate_ids.update(scene.objects.keys())
            scene.update_snapshots(
                obj_ids=candidate_ids,
                min_detection=int(
                    getattr(cfg, "snapshot_min_detection", getattr(cfg, "min_detection", 2))
                ),
            )
        else:
            scene.update_snapshots(
                obj_ids=set(),
                min_detection=int(
                    getattr(cfg, "snapshot_min_detection", getattr(cfg, "min_detection", 2))
                ),
            )
    except Exception as exc:
        logging.warning(f"scene.update_snapshots failed at step {cnt_step}: {exc}")

    _refresh_memory_from_scene_snapshots(
        scene=scene, tsdf_planner=tsdf_planner, app_state=app_state
    )

    tsdf_planner.update_frontier_map(
        pts=current_pts,
        cfg=cfg.planner if hasattr(cfg, "planner") else cfg,
        scene=scene,
        cnt_step=cnt_step,
        save_frontier_image=cfg.save_visualization,
        eps_frontier_dir=eps_frontier_dir,
    )

    return current_step_observations


def _refresh_memory_from_scene_snapshots(
    scene: Scene, tsdf_planner: TSDFPlanner, app_state: AppState
) -> None:
    """Scene 스냅샷과 observation memory를 동기화합니다."""
    if not hasattr(scene, "snapshots") or scene.snapshots is None:
        return

    existing_memory: Dict[str, AnalyzedObservation] = app_state.observation_memory
    updated_memory: Dict[str, AnalyzedObservation] = {}

    for snapshot_name, snapshot in scene.snapshots.items():
        obs_meta = scene.all_observations.get(snapshot_name)
        if not obs_meta:
            continue

        rgb = obs_meta.get("rgb")
        if rgb is None:
            continue

        map_pos_arr = np.asarray(snapshot.obs_point)
        if map_pos_arr.size < 2:
            continue
        map_pos = (int(map_pos_arr[0]), int(map_pos_arr[1]))
        angle = float(obs_meta.get("angle", 0.0))

        key_objects: List[str] = []
        object_infos: List[Dict[str, Any]] = []
        for obj_id, confidence in snapshot.full_obj_list.items():
            obj_meta = scene.objects.get(obj_id)
            if not obj_meta:
                continue
            class_name = obj_meta.get("class_name")
            if class_name:
                key_objects.append(class_name)
            bbox = obj_meta.get("bbox")
            if bbox is None or not hasattr(bbox, "center"):
                continue
            center_habitat = np.asarray(bbox.center)
            if center_habitat.size < 3:
                continue
            center_voxel = tsdf_planner.habitat2voxel(center_habitat)[:2]
            object_infos.append(
                {
                    "object_id": obj_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "center_habitat": center_habitat.tolist(),
                    "center_voxel": (
                        int(center_voxel[0]),
                        int(center_voxel[1]),
                    ),
                }
            )
        key_objects = list(dict.fromkeys(key_objects))

        existing_entry = existing_memory.get(snapshot_name)
        if existing_entry:
            existing_entry.map_pos = map_pos
            existing_entry.angle = angle
            existing_entry.rgb = rgb
            merged_keys = list(dict.fromkeys(existing_entry.key_objects + key_objects))
            existing_entry.key_objects = merged_keys
            existing_entry.object_infos = object_infos
            updated_memory[snapshot_name] = existing_entry
            continue

        descriptors: List[str] = []
        vlm_likelihood: float = 0.0
        parsed_objects: List[str] = []
        try:
            descriptors, vlm_likelihood, parsed_objects = (
                app_state.desc_extractor.analyze_scene_for_goal(
                    rgb=rgb,
                    goal="",
                    key_objects=key_objects,
                )
            )
        except Exception as exc:
            logging.warning(
                f"Descriptor extraction for snapshot {snapshot_name} failed: {exc}"
            )

        if not parsed_objects:
            parsed_objects = key_objects
        if not descriptors:
            descriptors = key_objects[: app_state.desc_extractor.n_descriptors]

        entry = AnalyzedObservation(
            name=snapshot_name,
            map_pos=map_pos,
            angle=angle,
            descriptors=descriptors,
            key_objects=list(dict.fromkeys((parsed_objects or []) + key_objects)),
            rgb=rgb,
            vlm_likelihood=vlm_likelihood,
            object_infos=object_infos,
        )
        updated_memory[snapshot_name] = entry

    app_state.observation_memory = updated_memory


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
    memory_candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    프론티어와 메모리 후보를 통합해 다음 탐색 대상을 선정합니다.
    - 1차 기준: SigLIP ITM 점수
    - 2차 기준: 탐사 점수(프론티어 전용)로 tie-break
    """
    cfg = app_state.cfg
    score_cache = episode_context.get("frontier_score_cache", {})
    subtask_obs_cache = episode_context.get("subtask_observation_cache", {})

    scored_candidates: List[Dict[str, Any]] = []
    scored_candidates_all: List[Dict[str, Any]] = []

    memory_threshold = float(
        getattr(app_state.cfg, "memory_navigation_threshold", 0.7)
    )

    # Memory candidates (이미 점수가 계산되어 있음)
    for mem in memory_candidates:
        if mem["combined_score"] < memory_threshold:
            logging.info(
                "Skipping memory candidate '%s' due to low ITM score %.3f < %.3f.",
                mem["observation"].name,
                mem["combined_score"],
                memory_threshold,
            )
            continue
        entry = {
            "type": "memory",
            "memory": mem,
            "semantic_score": mem["combined_score"],
            "score_itm": mem["combined_score"],
            "score_itm_primary": mem["primary_score"],
            "score_itm_secondary": mem["secondary_score"],
            "exploration_score": 0.0,
        }
        scored_candidates.append(entry)
        scored_candidates_all.append(entry)

    # Frontier candidates
    frontiers_with_data: List[Dict[str, Any]] = []
    for f in tsdf_planner.frontiers:
        if f.source_observation_name and f.source_observation_name in subtask_obs_cache:
            source_obs = subtask_obs_cache[f.source_observation_name]
            frontiers_with_data.append(
                {
                    "frontier": f,
                    "rgb": source_obs["rgb"],
                    "key_objects": source_obs.get("key_objects", []),
                    "descriptors": source_obs.get("descriptors"),
                }
            )

    cur_voxel = tsdf_planner.normal2voxel(pos_habitat_to_normal(current_pts))
    primary_targets_ctx: List[str] = (
        [target_objects[0]] if target_objects else []
    )
    secondary_targets_ctx = [
        str(t).strip()
        for t in episode_context.get("secondary_targets", [])
        if isinstance(t, str) and str(t).strip()
    ]

    for data in frontiers_with_data:
        f = data["frontier"]
        obs_cache_entry = None
        if f.source_observation_name:
            obs_cache_entry = episode_context["subtask_observation_cache"].get(
                f.source_observation_name
            )

        cached_scores = score_cache.get(f.frontier_id)
        if cached_scores is not None:
            logging.info(f"Cache HIT for Frontier {f.frontier_id}.")
            descriptors = cached_scores.get("descriptors", [])
            if descriptors and not data.get("descriptors"):
                data["descriptors"] = descriptors
            if obs_cache_entry is not None and descriptors:
                obs_cache_entry["descriptors"] = descriptors
            parsed_objects = cached_scores.get("parsed_objects")
            if parsed_objects is not None:
                data["key_objects"] = parsed_objects
                if obs_cache_entry is not None:
                    obs_cache_entry["key_objects"] = parsed_objects
            vlm_likelihood_score = cached_scores.get("vlm_likelihood", 0.0)
            primary_score = cached_scores.get("score_itm_primary", 0.0)
            secondary_score = cached_scores.get("score_itm_secondary", 0.0)
            combined_score = cached_scores.get("combined_itm", primary_score)
        else:
            descriptors = data.get("descriptors")
            if descriptors is None and obs_cache_entry is not None:
                descriptors = obs_cache_entry.get("descriptors")
            if descriptors is None:
                descriptors = []

            parsed_objects = data.get("key_objects", [])
            vlm_likelihood_score = 0.0

            logging.info(
                f"Cache MISS for Frontier {f.frontier_id}. Performing descriptor analysis."
            )
            try:
                if not descriptors:
                    descriptors, vlm_likelihood_score, parsed_objects = (
                        app_state.desc_extractor.analyze_scene_for_goal(
                            rgb=data["rgb"],
                            goal=goal,
                            key_objects=data["key_objects"],
                        )
                    )
                    data["descriptors"] = descriptors
                    if parsed_objects:
                        data["key_objects"] = parsed_objects
                    if obs_cache_entry is not None:
                        obs_cache_entry["descriptors"] = descriptors
                        if parsed_objects:
                            obs_cache_entry["key_objects"] = parsed_objects
            except Exception as exc:
                logging.warning(
                    f"Descriptor extraction failed for frontier {f.frontier_id}: {exc}"
                )
                descriptors = []
                vlm_likelihood_score = 0.0

            combined_score, primary_score, secondary_score = app_state.desc_extractor.compute_itm_scores(
                primary_targets_ctx,
                secondary_targets_ctx,
                descriptors,
            )
            score_cache[f.frontier_id] = {
                "descriptors": descriptors,
                "vlm_likelihood": vlm_likelihood_score,
                "score_itm_primary": primary_score,
                "score_itm_secondary": secondary_score,
                "combined_itm": combined_score,
                "parsed_objects": parsed_objects,
            }

        unexplored_volume = tsdf_planner.get_unexplored_volume_from_frontier(f)
        normalized_volume_score = np.tanh(unexplored_volume / 1000.0)
        distance = (
            np.linalg.norm(cur_voxel[:2] - f.position) * tsdf_planner._voxel_size
        )
        exploration_score = 0.25 * (normalized_volume_score + (1.0 / (1.0 + distance)))

        semantic_score = (cfg.w_itm_score * combined_score) + (
            cfg.w_vlm_score * vlm_likelihood_score
        )

        entry = {
            "type": "frontier",
            "frontier": f,
            "semantic_score": semantic_score,
            "score_itm": combined_score,
            "score_itm_primary": primary_score,
            "score_itm_secondary": secondary_score,
            "vlm_likelihood": vlm_likelihood_score,
            "exploration_score": exploration_score,
            "source_observation_name": f.source_observation_name,
        }
        scored_candidates.append(entry)
        scored_candidates_all.append(entry)

    episode_context["frontier_score_cache"] = score_cache

    if not scored_candidates:
        return None, None

    scored_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)
    best_candidate = scored_candidates[0]

    if best_candidate["semantic_score"] < cfg.low_score_threshold:
        frontier_only = [
            c for c in scored_candidates if c["type"] == "frontier"
        ]
        if frontier_only:
            frontier_only.sort(key=lambda x: x["exploration_score"], reverse=True)
            best_candidate = frontier_only[0]
            logging.info(
                "All scores below threshold %.2f. Using exploration to select frontier %s.",
                cfg.low_score_threshold,
                best_candidate["frontier"].frontier_id,
            )
        else:
            logging.info(
                "All scores below threshold but no frontiers available; using top memory candidate."
            )
    else:
        top_score = scored_candidates[0]["semantic_score"]
        tie_candidates = [
            c
            for c in scored_candidates
            if top_score - c["semantic_score"] < cfg.tie_breaking_threshold
        ]
        if len(tie_candidates) > 1:
            logging.info(
                "%d candidates in semantic tie. Using exploration score to break ties.",
                len(tie_candidates),
            )
            best_in_tie = max(
                tie_candidates,
                key=lambda x: x["semantic_score"] + x["exploration_score"],
            )
            if best_in_tie is not best_candidate:
                logging.info(
                    "Tie-breaker selected %s over previous best.",
                    "frontier"
                    if best_in_tie["type"] == "frontier"
                    else f"memory {best_in_tie['memory']['observation'].name}",
                )
            best_candidate = best_in_tie

    logging.info(
        "Chosen candidate: %s | Semantic %.3f (ITM %.3f, Exploration %.3f)",
        (
            f"frontier_{best_candidate['frontier'].frontier_id}"
            if best_candidate["type"] == "frontier"
            else f"memory_{best_candidate['memory']['observation'].name}"
        ),
        best_candidate["semantic_score"],
        best_candidate["score_itm"],
        best_candidate["exploration_score"],
    )

    candidates_info = None
    if app_state.cfg.save_visualization:
        candidates_log = []
        scores_for_log = {}

        for entry in scored_candidates_all:
            if entry["type"] == "frontier":
                frontier = entry["frontier"]
                obs_name = entry.get("source_observation_name")
                cache_entry = episode_context["subtask_observation_cache"].get(obs_name)
                if cache_entry is None:
                    continue
                name = f"frontier_{frontier.frontier_id}"
                candidates_log.append(
                    {
                        "type": "frontier",
                        "rgb": cache_entry["rgb"],
                        "name": name,
                    }
                )
            else:
                mem_obs = entry["memory"]["observation"]
                name = f"memory_{mem_obs.name}"
                candidates_log.append(
                    {
                        "type": "memory",
                        "rgb": mem_obs.rgb,
                        "name": name,
                    }
                )
            scores_for_log[name] = entry["semantic_score"]

        chosen_name_for_log = (
            f"frontier_{best_candidate['frontier'].frontier_id}"
            if best_candidate["type"] == "frontier"
            else f"memory_{best_candidate['memory']['observation'].name}"
        )
        caption = (
            f"Goal: {original_goal_str}\n"
            f"Chosen: {chosen_name_for_log} | Score: {best_candidate['semantic_score']:.2f}\n"
            f"ITM: {best_candidate['score_itm']:.2f}"
            + (
                f", Exploration: {best_candidate['exploration_score']:.2f}"
                if best_candidate["type"] == "frontier"
                else ""
            )
        )
        candidates_info = {
            "save_dir": selection_dir,
            "candidates": candidates_log,
            "chosen_name": chosen_name_for_log,
            "scores": scores_for_log,
            "caption": caption,
            "filename_prefix": "",
        }

    return best_candidate, candidates_info


def get_relevant_observations(
    frontier: Frontier,
    observations: List[Dict[str, Any]],
    angle_threshold_deg: float,
) -> List[Dict[str, Any]]:
    """프론티어와 각도 차이가 작은 관측을 필터링합니다."""
    relevant_obs = []
    frontier_angle = np.arctan2(frontier.orientation[1], frontier.orientation[0])
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    for obs_data in observations:
        obs_angle = obs_data.get("angle")
        if obs_angle is None:
            continue

        angle_diff = abs(frontier_angle - obs_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        if angle_diff <= angle_threshold_rad:
            relevant_obs.append(obs_data)

    logging.debug(
        "Frontier %s: Found %d relevant observations.",
        frontier.frontier_id,
        len(relevant_obs),
    )
    return relevant_obs


def main(
    cfg_file: str,
    start_ratio: float,
    end_ratio: float,
    split: int,
    scene_id: Optional[str] = None,
):
    """메인 실행 함수."""
    app_state = setup(cfg_file, start_ratio, end_ratio, split)
    run_evaluation(app_state, start_ratio, end_ratio, split, scene_id)
    logging.info("All scenes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--cfg_file",
        help="cfg file path",
        default="cfg/eval_mzson_mem.yaml",
        type=str,
    )
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument(
        "--scene_id",
        help="Run evaluation for a specific scene ID",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    if args.scene_id == "" or args.scene_id == "${input:sceneId}":
        args.scene_id = None

    logging.info("***** Running Memory-Aware MZSON methodology *****")
    main(args.cfg_file, args.start_ratio, args.end_ratio, args.split, args.scene_id)

