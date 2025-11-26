from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import torch


@dataclass
class Frontier:
    """Frontier class for frontier-based exploration."""

    position: np.ndarray  # integer position in voxel grid
    orientation: np.ndarray  # directional vector of the frontier in float
    region: (
        np.ndarray
    )  # boolean array of the same shape as the voxel grid, indicating the region of the frontier
    frontier_id: (
        int  # unique id for the frontier to identify its region on the frontier map
    )
    source_observation_name: Optional[str] = None
    cam_pose: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    image_path: str = None # image_path로 이름 변경
    
    # 프론티어 방향을 직접 바라본 관측 이미지 (Numpy array)
    # repr=False, eq=False를 통해 dataclass의 기본 출력 및 비교 연산에서 제외
    observation_image: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    observation_depth: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    observation_cam_pose: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    target_detected: bool = (
        False  # whether the target object is detected in the snapshot, only used when generating data
    )
    feature: torch.Tensor = (
        None  # the image feature of the snapshot, not used when generating data
    )
    group_id: Optional[int] = None

    # Calculated geometric properties, to be filled in later
    region_size: int = 0
    distance: float = 0.0
    visibility_score: float = 1.0

    def __eq__(self, other):
        if not isinstance(other, Frontier):
            raise TypeError("Cannot compare Frontier with non-Frontier object.")
        return np.array_equal(self.region, other.region)


