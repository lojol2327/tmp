from __future__ import annotations

import os
import json
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import glob
import logging

import numpy as np
import habitat_sim
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as patheffects

# Forward declarations to avoid circular imports
class Scene:
    pass
class TSDFPlanner:
    pass


class MZSONLogger:
    """Logger for mzson experiments, mirroring useful features from goatbench logger.

    - Tracks success and SPL statistics
    - Persists intermediate aggregated results
    - Manages episode/subtask directories
    - Adds mzson-specific hooks to record frontier candidates, descriptors, and selections
    """

    def __init__(
        self,
        output_dir: str,
        start_ratio: float,
        end_ratio: float,
        split: int,
        voxel_size: float,
    ) -> None:
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        os.makedirs(self.output_dir, exist_ok=True)

        # Statistics stores
        def _load_or_default(path: str, default):
            if os.path.exists(path):
                with open(path, "rb" if path.endswith(".pkl") else "r") as f:
                    return pickle.load(f) if path.endswith(".pkl") else json.load(f)
            return default

        self.success_by_snapshot = _load_or_default(
            os.path.join(self.output_dir, f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.success_by_distance = _load_or_default(
            os.path.join(self.output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.spl_by_snapshot = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.spl_by_distance = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"),
            {},
        )
        self.success_by_task = _load_or_default(
            os.path.join(self.output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl"),
            defaultdict(list),
        )
        self.spl_by_task = _load_or_default(
            os.path.join(self.output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"),
            defaultdict(list),
        )
        self.n_filtered_snapshots_list = _load_or_default(
            os.path.join(self.output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )
        self.n_total_snapshots_list = _load_or_default(
            os.path.join(self.output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )
        self.n_total_frames_list = _load_or_default(
            os.path.join(self.output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"),
            {},
        )

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.split = split

        # Episode/Subtask state
        self.episode_dir: str | None = None
        self.subtask_object_observe_dir: str | None = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    # ---------- persistence ----------
    def save_results(self) -> None:
        def _dump(obj, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb" if path.endswith(".pkl") else "w") as f:
                pickle.dump(obj, f) if path.endswith(".pkl") else json.dump(obj, f, indent=4)

        _dump(
            self.success_by_snapshot,
            os.path.join(self.output_dir, f"success_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.success_by_distance,
            os.path.join(self.output_dir, f"success_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_snapshot,
            os.path.join(self.output_dir, f"spl_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_distance,
            os.path.join(self.output_dir, f"spl_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.success_by_task,
            os.path.join(self.output_dir, f"success_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.spl_by_task,
            os.path.join(self.output_dir, f"spl_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl"),
        )
        _dump(
            self.n_filtered_snapshots_list,
            os.path.join(self.output_dir, f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )
        _dump(
            self.n_total_snapshots_list,
            os.path.join(self.output_dir, f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )
        _dump(
            self.n_total_frames_list,
            os.path.join(self.output_dir, f"n_total_frames_{self.start_ratio}_{self.end_ratio}_{self.split}.json"),
        )

    # ---------- aggregation ----------
    def aggregate_results(self) -> None:
        """여러 조각으로 나뉜 실험 결과 파일들을 하나로 종합합니다."""
        filenames_to_merge = [
            "success_by_snapshot", "spl_by_snapshot",
            "success_by_distance", "spl_by_distance",
        ]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(os.path.join(self.output_dir, f"{filename}_*.pkl"))
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    all_results.update(pickle.load(f))
            
            if all_results:
                logging.info(f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}, len: {len(all_results)}")
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        filenames_to_merge = ["success_by_task", "spl_by_task"]
        for filename in filenames_to_merge:
            all_results = defaultdict(list)
            all_results_paths = glob.glob(os.path.join(self.output_dir, f"{filename}_*.pkl"))
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    separate_stat = pickle.load(f)
                    for task_name, task_res in separate_stat.items():
                        all_results[task_name] += task_res
            
            for task_name, task_res in all_results.items():
                logging.info(f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}, len: {len(task_res)}")
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        json_files_to_merge = ["n_filtered_snapshots", "n_total_snapshots", "n_total_frames"]
        for filename in json_files_to_merge:
            all_results = {}
            all_results_paths = glob.glob(os.path.join(self.output_dir, f"{filename}_*.json"))
            for results_path in all_results_paths:
                with open(results_path, "r") as f:
                    all_results.update(json.load(f))
            if all_results:
                logging.info(f"Average number of {filename}: {np.mean(list(all_results.values()))}")
            with open(os.path.join(self.output_dir, f"{filename}.json"), "w") as f:
                json.dump(all_results, f, indent=4)


    # ---------- episode/subtask management ----------
    def init_episode(self, scene_id, episode_id):
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.episode_dir = os.path.join(
            self.output_dir, scene_id
        )  # _ep_{episode_id} 제거
        os.makedirs(self.episode_dir, exist_ok=True)

        self.subtask_results = []
        self.cumulative_stats = {"success": 0, "spl": 0, "subtask_count": 0}

    def init_subtask(
        self,
        subtask_id: str,
        goal_type: str,
        subtask_goal: Any,
        pts: np.ndarray,
        scene: Scene,
        tsdf_planner: TSDFPlanner,
    ) -> Dict[str, Any]:
        self.subtask_id = subtask_id
        # --- subtask를 위한 모든 디렉토리 생성 ---
        self.subtask_dir = os.path.join(
            self.episode_dir, f"subtask_{self.subtask_id}"
        )  # more intuitive naming

        # Clear and create subdirectories for the current subtask
        self.frontier_dir = os.path.join(self.subtask_dir, "frontier")
        self.visualization_dir = os.path.join(self.subtask_dir, "visualization")
        self.subtask_object_observe_dir = os.path.join(
            self.subtask_dir, "object_observations"
        )
        self.selection_dir = os.path.join(self.subtask_dir, "selection") # Unified selection folder

        # 기존 디렉토리 정리
        if os.path.exists(self.subtask_dir):
            try:
                import shutil
                shutil.rmtree(self.subtask_dir)
            except Exception:
                pass
        
        os.makedirs(self.subtask_dir, exist_ok=True)
        os.makedirs(self.subtask_object_observe_dir, exist_ok=True)
        os.makedirs(self.frontier_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.selection_dir, exist_ok=True) # Create the unified folder

        # --- Reset per-subtask state ---
        self.pts_voxels = np.empty((0, 2))
        # Initialize with starting position (critical for distance calculation!)
        init_pts_voxel = tsdf_planner.habitat2voxel(pts)[:2]
        self.pts_voxels = np.vstack([self.pts_voxels, init_pts_voxel])
        self.subtask_explore_dist = 0.0

        # --- Parse Goal Information (from logger_goatbench.py) ---
        goal_category = subtask_goal[0]["object_category"]
        goal_obj_ids = [int(x["object_id"].split("_")[-1]) for x in subtask_goal]
        goal_positions = [x["position"] for x in subtask_goal]
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]
        
        viewpoints = [
            view_point["agent_state"]["position"]
            for goal in subtask_goal
            for view_point in goal["view_points"]
        ]

        # Calculate ground-truth distance for SPL
        all_distances = []
        for viewpoint in viewpoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = viewpoint
            found_path = scene.pathfinder.find_path(path)
            if not found_path:
                all_distances.append(np.inf)
            else:
                all_distances.append(path.geodesic_distance)
        gt_subtask_explore_dist = min(all_distances) + 1e-6

        # --- Construct Metadata Dictionary ---
        subtask_metadata = {
            "subtask_id": subtask_id,
            "question": None,
            "image": None,
            "goal_obj_ids": goal_obj_ids,
            "class": goal_category,
            "goal_positions_voxel": goal_positions_voxel,
            "goal_type": goal_type,
            "viewpoints": viewpoints,
            "gt_subtask_explore_dist": gt_subtask_explore_dist,
            "goal": None, # Will be populated below
        }

        # Format question and goal based on type
        if goal_type == "object":
            subtask_metadata["question"] = f"Find the {goal_category}"
            subtask_metadata["goal"] = [goal_category]
        elif goal_type == "description":
            desc = subtask_goal[0]['lang_desc']
            subtask_metadata["question"] = f"Could you find the object exactly described as the '{desc}'?"
            subtask_metadata["goal"] = [desc]
        elif goal_type == "image":
            subtask_metadata["question"] = "Could you find the exact object captured at the center of the following image?"
            view_pos_dict = subtask_goal[0]["view_points"][0]["agent_state"]
            obs, _ = scene.get_observation(
                pts=view_pos_dict["position"], rotation=view_pos_dict["rotation"]
            )
            image_goal_path = os.path.join(self.subtask_object_observe_dir, "image_goal.png")
            plt.imsave(image_goal_path, obs["color_sensor"])
            subtask_metadata["image"] = Image.open(image_goal_path)
            subtask_metadata["goal"] = subtask_metadata["image"]

        self.current_subtask_metadata = subtask_metadata
        return self.current_subtask_metadata

    def log_step(self, pts_voxel: np.ndarray) -> None:
        if self.pts_voxels.size == 0:
            self.pts_voxels = np.expand_dims(pts_voxel, axis=0)
        else:
            self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
            
        if len(self.pts_voxels) >= 2:
            self.subtask_explore_dist += (
                np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
            )

    def get_frontier_dir(self) -> str | None:
        """현재 서브태스크의 프론티어 디렉토리 경로를 반환합니다."""
        return getattr(self, "frontier_dir", None)

    def save_frontier_visualization(
        self,
        global_step: int,
        subtask_id: str,
        tsdf_planner, # TSDFPlanner,
        max_point_choice, # Union[Frontier, SnapShot],
        caption: str,
    ):
        """에이전트가 고려한 모든 프론티어와 최종 선택지를 한 이미지로 저장합니다."""
        assert self.subtask_dir is not None
        frontier_video_path = os.path.join(self.subtask_dir, "frontier_selection")
        os.makedirs(frontier_video_path, exist_ok=True)
        
        source_frontier_dir = self.get_frontier_dir()
        if not source_frontier_dir: return

        frontiers = tsdf_planner.frontiers
        num_images = len(frontiers)
        if num_images == 0: return

        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if i < num_images:
                    frontier = frontiers[i]
                    if not frontier.image_path: continue
                    img_path = os.path.join(source_frontier_dir, frontier.image_path)
                    if not os.path.exists(img_path): continue
                    
                    img = mpimg.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    
                    if max_point_choice and max_point_choice.frontier_id == frontier.frontier_id:
                        axs[h_idx, w_idx].set_title("Chosen", color='r', fontsize=20)

        fig.suptitle(caption, fontsize=24)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(
            os.path.join(frontier_video_path, f"step_{global_step:04d}_{subtask_id}.png")
        )
        plt.close(fig)


    def save_candidates_visualization(
        self,
        candidates: List[Dict[str, Any]],
        chosen_name: str,
        save_dir: str,
        subtask_id: str,
        step: int,
        scores: Dict[str, float],
        caption: str = "Candidates",
        filename_prefix: str = "candidates_", # Receive the prefix
        exploration_scores: Optional[Dict[str, float]] = None,
        tie_break_applied: bool = False,
        active_group_id: Optional[int] = None,
        active_group_label: Optional[str] = None,
        group_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Saves a collage of candidate observations/frontiers with their scores.
        Highlights the chosen one.
        """
        # Save collage of all considered observations
        collage_path = os.path.join(
            save_dir, f"{filename_prefix}step_{step:04d}_{subtask_id}_candidates.png"
        )

        num_images = len(candidates)
        if num_images == 0: return

        side_length = int(np.ceil(np.sqrt(num_images)))
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        axs = np.atleast_1d(axs).flatten() # Ensure axs is always a flat array for easy iteration
        group_labels = group_labels or {}

        for i, ax in enumerate(axs):
            ax.axis("off")
            if i < num_images:
                candidate = candidates[i]
                img = candidate.get("rgb")
                name = candidate.get("name", f"candidate_{i}")
                score = scores.get(name, 0.0)
                exploration_score = None
                if exploration_scores:
                    exploration_score = exploration_scores.get(name)
                
                if img is not None:
                    ax.imshow(img)
                    subtitle_parts = [f"Score: {score:.3f}"]
                    if exploration_score is not None:
                        subtitle_parts.append(f"Expl: {exploration_score:.3f}")
                    if candidate.get("is_memory"):
                        subtitle_parts.append("MEM")
                    group_label = candidate.get("group_label")
                    if (not group_label) and candidate.get("type") == "frontier":
                        if name.startswith("frontier_"):
                            try:
                                fid = int(name.split("_")[1])
                                group_label = group_labels.get(fid)
                            except Exception:
                                group_label = None
                    if group_label:
                        subtitle_parts.append(f"Grp {group_label}")
                    title = f"{os.path.basename(name)}\n" + ", ".join(subtitle_parts)
                    
                    if name == chosen_name:
                        ax.set_title(title, color='r', fontsize=16, fontweight='bold')
                        # Add a red border
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(5)
                    else:
                        ax.set_title(title, fontsize=12)

                    group_id = candidate.get("group_id")
                    if group_id is not None and name != chosen_name:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('blue')
                            spine.set_linewidth(3)

        caption_lines = [caption]
        if tie_break_applied:
            caption_lines.append("Tie-break applied (semantic + exploration)")
        if active_group_label:
            caption_lines.append(f"Active frontier group: {active_group_label}")
        elif active_group_id is not None:
            caption_lines.append(f"Active frontier group: {active_group_id}")
        fig.suptitle("\n".join(caption_lines), fontsize=24)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = f"step_{step:04d}_{subtask_id}_candidates.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    def log_step_visualizations(
        self,
        global_step: int,
        subtask_id: str,
        subtask_metadata: Dict[str, Any],
        fig: plt.Figure,
        candidates_info: Dict[str, Any] | None = None,
        memory_targets: Optional[List[Tuple[int, int]]] = None,
    ):
        """A centralized function to save all visualizations for a single step."""
        # 1. Save the main top-down map visualization
        self.save_topdown_visualization(
            global_step,
            subtask_id,
            subtask_metadata,
            fig,
            memory_targets=memory_targets,
        )

        # 2. If there are candidates to visualize, save them as a collage
        if candidates_info and candidates_info.get("candidates"):
            subtask_id = subtask_metadata["subtask_id"]
            filename_prefix = candidates_info.get("filename_prefix", "candidates_")

            self.save_candidates_visualization(
                candidates_info["candidates"],
                candidates_info["chosen_name"],
                candidates_info["save_dir"],
                subtask_id,
                global_step,
                candidates_info["scores"],
                caption=candidates_info.get("caption", "Candidates"),
                filename_prefix=filename_prefix, # Pass the prefix
                exploration_scores=candidates_info.get("exploration_scores"),
                tie_break_applied=candidates_info.get("tie_break_applied", False),
                active_group_id=candidates_info.get("active_group_id"),
                active_group_label=candidates_info.get("active_group_label"),
                group_labels=candidates_info.get("group_labels"),
            )

    def save_topdown_visualization(
        self,
        global_step: int,
        subtask_id: str,
        subtask_metadata: Dict[str, Any],
        fig,
        memory_targets: Optional[List[Tuple[int, int]]] = None,
    ):
        """Top-down map에 에이전트 경로와 목표 지점을 그려서 저장합니다."""
        assert self.visualization_dir is not None
        
        ax1 = fig.axes[0]
        
        # 에이전트의 이동 경로 그리기
        ax1.plot(
            self.pts_voxels[:, 1], self.pts_voxels[:, 0], linewidth=1, color="white"
        )
        # 에이전트의 시작 지점 그리기
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # 목표 지점(들) 그리기
        for goal_pos_voxel in subtask_metadata["goal_positions_voxel"]:
            # 목표 지점을 테두리가 굵은 빨간색이고 속은 파란색인 별표로 표시
            ax1.scatter(
                goal_pos_voxel[1], 
                goal_pos_voxel[0], 
                facecolor='blue', 
                edgecolor='red', 
                linewidth=2, 
                s=350, 
                marker='p', 
                zorder=10
            )
            
            # 목표 지점 위에 "goal" 텍스트 추가 (진한 녹색, 약간 큰 폰트)
            ax1.text(
                goal_pos_voxel[1], 
                goal_pos_voxel[0] - 5, # 마커 바로 위에 위치하도록 y좌표 조정
                "goal", 
                color='darkgreen', 
                fontsize=14, 
                ha='center', # 수평 중앙 정렬
                va='bottom'  # 수직 하단 정렬
            )

        if memory_targets:
            for mem_pos in memory_targets:
                if mem_pos is None:
                    continue
                try:
                    ax1.scatter(
                        mem_pos[1],
                        mem_pos[0],
                        marker="*",
                        s=300,
                        facecolor="yellow",
                        edgecolor="black",
                        linewidth=1.5,
                        zorder=15,
                    )
                    ax1.text(
                        mem_pos[1],
                        mem_pos[0] + 3,
                        "memory",
                        color="yellow",
                        fontsize=12,
                        ha="center",
                        va="bottom",
                        zorder=16,
                    )
                except Exception as exc:
                    logging.warning(f"Failed to mark memory target on map: {exc}")

        fig.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, f"{subtask_id}_{global_step:04d}.png"))
        plt.close(fig)

    def save_success_visualization(
        self, subtask_id: str, subtask_metadata: Dict[str, Any], fig
    ):
        """성공 시 최종 agent 위치를 보여주는 visualization을 저장합니다."""
        assert self.visualization_dir is not None
        
        ax1 = fig.axes[0]
        
        # 에이전트의 이동 경로 그리기
        ax1.plot(
            self.pts_voxels[:, 1], self.pts_voxels[:, 0], linewidth=1, color="white"
        )
        # 에이전트의 시작 지점 그리기
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # 목표 지점(들) 그리기
        for goal_pos_voxel in subtask_metadata["goal_positions_voxel"]:
            # 목표 지점을 테두리가 굵은 빨간색이고 속은 파란색인 별표로 표시
            ax1.scatter(
                goal_pos_voxel[1], 
                goal_pos_voxel[0], 
                facecolor='blue', 
                edgecolor='red', 
                linewidth=2, 
                s=350, 
                marker='p', 
                zorder=10
            )
            
            # 목표 지점 위에 "goal" 텍스트 추가 (진한 녹색, 약간 큰 폰트)
            ax1.text(
                goal_pos_voxel[1], 
                goal_pos_voxel[0] - 5, # 마커 바로 위에 위치하도록 y좌표 조정
                "goal", 
                color='darkgreen', 
                fontsize=14, 
                ha='center', # 수평 중앙 정렬
                va='bottom'  # 수직 하단 정렬
            )

        fig.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, f"{subtask_id}_success.png"))
        plt.close(fig)

    # ---------- subtask result ----------
    def log_subtask_result(
        self,
        success_by_snapshot: bool,
        success_by_distance: bool,
        subtask_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_snapshots: int,
        n_total_snapshots: int,
        n_total_frames: int,
    ) -> None:
        self.success_by_snapshot[subtask_id] = 1.0 if success_by_snapshot else 0.0
        self.success_by_distance[subtask_id] = 1.0 if success_by_distance else 0.0

        # SPL (Success weighted by Path Length)
        def _spl(success: float) -> float:
            denom = max(gt_subtask_explore_dist, self.subtask_explore_dist, 1e-6)
            return success * gt_subtask_explore_dist / denom

        self.spl_by_snapshot[subtask_id] = _spl(self.success_by_snapshot[subtask_id])
        self.spl_by_distance[subtask_id] = _spl(self.success_by_distance[subtask_id])
        self.success_by_task.setdefault(goal_type, []).append(self.success_by_distance[subtask_id])
        self.spl_by_task.setdefault(goal_type, []).append(self.spl_by_distance[subtask_id])

        # snapshot/frame counts
        self.n_filtered_snapshots_list[subtask_id] = int(n_filtered_snapshots)
        self.n_total_snapshots_list[subtask_id] = int(n_total_snapshots)
        self.n_total_frames_list[subtask_id] = int(n_total_frames)

        # --- 누적 통계 출력 (베이스라인 기능 이식) ---
        logging.info(f"Subtask {subtask_id} finished, traversed {self.subtask_explore_dist:.2f}m")
        logging.info(f"SPL by distance: {self.spl_by_distance[subtask_id]:.4f}")

        if self.success_by_distance:
            logging.info(f"--- Cumulative Stats ---")
            logging.info(
                f"Success rate by distance: {100 * np.mean(np.asarray(list(self.success_by_distance.values()))):.2f}%"
            )
            logging.info(
                f"SPL by distance: {100 * np.mean(np.asarray(list(self.spl_by_distance.values()))):.2f}%"
            )
            for task_name, success_list in self.success_by_task.items():
                if success_list:
                    logging.info(
                        f"  - Success rate for '{task_name}': {100 * np.mean(np.asarray(success_list)):.2f}% ({len(success_list)} tasks)"
                    )
            for task_name, spl_list in self.spl_by_task.items():
                if spl_list:
                    logging.info(
                        f"  - SPL for '{task_name}': {100 * np.mean(np.asarray(spl_list)):.2f}% ({len(spl_list)} tasks)"
                    )

        # reset per-subtask trackers
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0
        self.frontier_dir = None
        self.visualization_dir = None
        self.subtask_dir = None

    # ---------- mzson specific hooks ----------
    def log_frontier_candidates(
        self,
        step: int,
        frontiers: List[Any],
        descriptors: List[Any],
        scores: List[float] | None = None,
    ) -> None:
        path = os.path.join(self.output_dir, f"candidates_{step:03d}.json")
        data = {
            "step": step,
            "num_frontiers": len(frontiers),
            "num_descriptors": len(descriptors),
            "scores": scores if scores is not None else [],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_selection(self, step: int, reason: str = "", **kwargs) -> None:
        path = os.path.join(self.output_dir, f"selection_{step:03d}.json")
        decision_data = kwargs  # Use kwargs as the decision dictionary
        with open(path, "w") as f:
            json.dump({"step": step, "decision": decision_data, "reason": reason}, f, indent=2)



