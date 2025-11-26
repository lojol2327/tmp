import logging
from typing import Optional

import numpy as np

from src.geom import get_proper_observe_point_with_pathfinder
from src.habitat import pos_habitat_to_normal, pos_normal_to_habitat
from src.tsdf_planner import TSDFPlanner, SnapShot


class TSDFPlannerMem(TSDFPlanner):
    """
    Memory-aware TSDF planner.

    Extends the base `TSDFPlanner` with the ability to advance toward memory targets
    even when the exact snapshot location is currently unreachable (e.g., occupied or
    outside the traversable island). The planner will move as far as possible toward
    the target, stepping one navigable voxel at a time along the direct line.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_target_voxel: Optional[np.ndarray] = None

    def _find_progress_voxel_towards(
        self,
        start_voxel: np.ndarray,
        target_voxel: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Returns the furthest navigable voxel along the straight line from start to target.
        If no progress is possible, returns None.
        """
        direction = target_voxel.astype(float) - start_voxel.astype(float)
        total_dist = np.linalg.norm(direction)
        if total_dist < 1e-6:
            return start_voxel.astype(int)

        steps = int(np.ceil(total_dist))
        steps = max(steps, 1)
        step_vec = direction / steps

        pos = start_voxel.astype(float)
        last_valid = None
        for _ in range(steps):
            pos += step_vec
            cand = np.round(pos).astype(int)
            if not self.check_within_bnds(cand):
                break
            if self.occupied is not None and self.occupied[cand[0], cand[1]]:
                break
            if self.island is not None and not self.island[cand[0], cand[1]]:
                break
            last_valid = cand.copy()

        return last_valid

    def set_target_point_from_observation(
        self,
        target_map_pos: np.ndarray,
        current_pts: Optional[np.ndarray] = None,
        object_voxel: Optional[np.ndarray] = None,
        pathfinder=None,
    ) -> bool:
        """
        Overrides the base method to allow partial progress toward the memory target.
        If the exact snapshot voxel is not navigable, the planner will step toward it
        as far as possible within the current free space.
        """
        target_arr = np.asarray(target_map_pos).flatten()
        if target_arr.size < 2 and object_voxel is None:
            logging.error(
                "set_target_point_from_observation: invalid target_map_pos %s",
                target_map_pos,
            )
            return False
        if object_voxel is not None:
            target_voxel = np.round(np.asarray(object_voxel)[:2]).astype(int)
        else:
            target_voxel = np.round(target_arr[:2]).astype(int)

        nav_voxel = target_voxel.astype(int)
        if (
            pathfinder is not None
            and current_pts is not None
            and self.unoccupied is not None
            and self.occupied is not None
        ):
            try:
                target_normal = (
                    nav_voxel.astype(float) * self._voxel_size + self._vol_origin[:2]
                )
                target_normal3 = np.array(
                    [target_normal[0], target_normal[1], self.floor_height]
                )
                target_habitat = pos_normal_to_habitat(target_normal3)
                candidate_habitat = get_proper_observe_point_with_pathfinder(
                    target_habitat, pathfinder, height=self.floor_height
                )
                if candidate_habitat is not None:
                    candidate_voxel = self.habitat2voxel(candidate_habitat)[:2]
                    candidate_voxel = np.round(candidate_voxel).astype(int)
                    if self.check_within_bnds(candidate_voxel):
                        is_on_island = (
                            self.island is None
                            or self.island[candidate_voxel[0], candidate_voxel[1]]
                        )
                        is_free = (
                            self.occupied is None
                            or not self.occupied[
                                candidate_voxel[0], candidate_voxel[1]
                            ]
                        )
                        if is_on_island and is_free:
                            nav_voxel = candidate_voxel
            except Exception as exc:
                logging.warning(
                    "set_target_point_from_observation: failed to adjust memory target "
                    "with pathfinder due to %s. Proceeding with raw target.",
                    exc,
                )

        obs_point = (
            np.round(target_arr[:2]).astype(int)
            if target_arr.size >= 2
            else nav_voxel.copy()
        )

        snapshot = SnapShot(
            image="observation_target",
            color=(0, 255, 0),
            obs_point=obs_point,
            position=np.array(target_map_pos),
        )
        snapshot.memory_target = target_voxel.astype(int)
        snapshot.position = target_voxel.astype(float)

        self.target_point = nav_voxel
        self.max_point = snapshot
        self._memory_target_voxel = snapshot.memory_target
        return True

    def set_next_navigation_point(
        self,
        choice,
        pts,
        objects,
        cfg,
        pathfinder,
        random_position=False,
        observe_snapshot=True,
    ):
        """
        Continues stepping toward the stored memory target. Falls back to the base
        implementation for non-memory choices.
        """
        if isinstance(choice, SnapShot) and hasattr(choice, "memory_target"):
            cur_normal = pos_habitat_to_normal(np.asarray(pts))
            start_voxel = self.normal2voxel(cur_normal)[:2]
            target_voxel = choice.memory_target

            progress_voxel = self._find_progress_voxel_towards(start_voxel, target_voxel)
            if progress_voxel is None:
                logging.warning(
                    "set_next_navigation_point: unable to make progress toward memory target %s.",
                    target_voxel,
                )
                return False

            self.target_point = progress_voxel.astype(int)
            return True

        return super().set_next_navigation_point(
            choice=choice,
            pts=pts,
            objects=objects,
            cfg=cfg,
            pathfinder=pathfinder,
            random_position=random_position,
            observe_snapshot=observe_snapshot,
        )

