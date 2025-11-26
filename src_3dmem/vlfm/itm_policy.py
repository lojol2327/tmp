# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from openai import OpenAI
from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.mapping.new_value_map import NewValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.utils.geometry_utils import within_fov_cone, extract_yaw

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "14182")))
        self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2ITM_PORT", "14185")))
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._new_value_map: NewValueMap = NewValueMap(
            value_channels=1,
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._new_value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best frontier : {best_frontier}, Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action
    
    def _look_around(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best frontier : {best_frontier}, Best value: {best_value*100:.2f}%") # for simple test
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action
    
    def _is_reinitialize_required(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> bool:
        """
        Check if reinitialization is required based on the condition:
        - The distance from the last initialized position.
        - There is any frontier whose value is within 0.04 of the best frontier's value.
        - That frontier is directed over 90 degrees away from the robot's current position
        compared to the best frontier.
        """
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            return False
        
        robot_xy = self._observations_cache["robot_xy"]
        # distance_from_last_init = np.linalg.norm(robot_xy - self._last_initialized_pos)
        # if self._num_steps > 12 and distance_from_last_init < 2.0: # 기본 4.0
        #     return False
        """
        """
        time_spent_from_last_init = self._num_steps - self._last_initialized_step
        if not self._num_steps <= 12 and time_spent_from_last_init < 30:
            return False
        
        """
        """
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        best_frontier = sorted_pts[0]
        best_value = sorted_values[0]
        
        for frontier, value in zip(sorted_pts[1:], sorted_values[1:]):  # Skip the best frontier
            if abs(value - best_value) <= 0.25: #0.25
                # Calculate the angle difference
                vector_to_best = best_frontier - robot_xy
                vector_to_current = frontier - robot_xy
                angle_diff = self._angle_between_vectors(vector_to_best, vector_to_current)
                if angle_diff > 45:  # Frontier is in the opposite direction
                    print("Reinitialization required: Significant frontier found in opposite direction.")
                    return True
                # return True
        return False

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate the angle between two vectors in degrees.
        """
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)  # Clip for numerical stability
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        robot_yaw = self._observations_cache["robot_heading"]
        best_frontier_idx = None
        
        """
        """
        def is_in_fov(frontier: np.ndarray, robot_xy: np.ndarray) -> bool:
            direction_to_frontier = frontier - robot_xy
            direction_to_frontier /= np.linalg.norm(direction_to_frontier)

            frontier_angle = np.arctan2(direction_to_frontier[1], direction_to_frontier[0])
            angle_diff = np.abs(np.arctan2(np.sin(robot_yaw - frontier_angle), np.cos(robot_yaw - frontier_angle)))
            
            return angle_diff < np.pi / 2
        
        # Apply temporary decrease for back-facing frontiers
        adjusted_values = sorted_values.copy()
        for idx, frontier in enumerate(sorted_pts):
            direction_to_frontier = frontier - robot_xy
            direction_to_frontier /= np.linalg.norm(direction_to_frontier)

            frontier_angle = np.arctan2(direction_to_frontier[1], direction_to_frontier[0])
            angle_diff = np.abs(np.arctan2(np.sin(robot_yaw - frontier_angle), np.cos(robot_yaw - frontier_angle)))

            adjusted_values[idx] = adjusted_values[idx] * np.cos(angle_diff / 2) #(1 + np.cos(angle_diff / 2)) / 2
        
        top_two_values = tuple(adjusted_values[:2])
        ##top_two_values = tuple(sorted_values[:2]) # VLFM

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = adjusted_values[curr_index]
                ##curr_value = sorted_values[curr_index] # VLFM
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = adjusted_values[best_frontier_idx] # sorted->adjusted
        ##best_value = sorted_values[best_frontier_idx] # VLFM
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )
        
    def _select_frontier_from_description(self, frontiers, direction):
        """
        Selects the best frontier point based on descriptions, scores, and FOV constraints.

        Returns:
            tuple: A tuple containing:
            - np.ndarray: The selected 2D frontier point.
            - float: The best score.
        """
        cropped_images = []
        for i, image in enumerate(self._images):
            # Check image dimensions
            if image.shape[1] != 640 or image.shape[0] != 480:
                raise ValueError(f"Image {i+1} has invalid dimensions {image.shape}. Expected 640x480.")

            # Crop the center part (480x480)
            x_center = image.shape[1] // 2
            x_start = x_center - 480 // 2
            x_end = x_center + 480 // 2
            cropped_image = image[:, x_start:x_end]  # Crop center horizontally
            cropped_images.append(cropped_image)
        
        # Step 1: Get yaws and direction
        yaws = []
        for i in range(len(self._observations_list)):
            tf_camera_to_episodic = self._observations_list[i][2]
            yaw = extract_yaw(tf_camera_to_episodic)
            yaws.append(yaw)
        wide = 2048
        #direction_yaw = yaws[0] - (direction - 939) / 2048 * 2 * np.pi
        # low resolution
        direction_yaw = yaws[0] - (direction - wide*0.458) / wide * 2 * np.pi
            
        # Step 2: Calculate scores and get the best index
        scores = []
        for i in range(len(cropped_images)):
            temp_scores = []
            image = cropped_images[i]
            for description in self._descriptions:
                score = self._itm.cosine(image, description)
                temp_scores.append(score)
                # run BLIP-2
            direction_diff = np.mod(yaws[i] - direction_yaw + np.pi, 2 * np.pi) - np.pi
            average_score = sum(temp_scores) / len(temp_scores) * np.cos(direction_diff/2)
            
            scores.append(average_score)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_yaw = yaws[best_idx]

        # Step 3: Define cone parameters
        robot_xy = self._observations_cache["robot_xy"]
        """
        """
        cone_fov = np.radians(30)  # 30 degrees to radians
        cone_range = 10.0  # Sufficiently large value as specified

        # Step 4: Filter frontiers within FOV cone
        if len(frontiers) > 1:
            best_frontiers = within_fov_cone(
                cone_origin=robot_xy,
                cone_angle=best_yaw,
                cone_fov=cone_fov,
                cone_range=cone_range,
                points=frontiers,
            )
        else:
            best_frontiers = np.empty((0, frontiers.shape[1]))
            

        # Step 5: Choose the closest frontier or calculate angular closest
        if len(best_frontiers) > 0:
            # Find the closest frontier to robot_xy
            distances = np.linalg.norm(best_frontiers - robot_xy, axis=1)
            selected_frontier = best_frontiers[np.argmin(distances)]
            print("Best frontier found in cone.")
        else:
            # No frontier in the cone; calculate angular difference
            directions = frontiers - robot_xy
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            angle_diffs = np.abs(np.mod(angles - best_yaw + np.pi, 2 * np.pi) - np.pi)
            selected_frontier = frontiers[np.argmin(angle_diffs)]
            print("Angular closest frontier selected.")
            print(f"Angle: {np.min(angle_diffs)*180/np.pi:.2f}, Distance: {np.linalg.norm(directions[np.argmin(angle_diffs)])}")

        return selected_frontier, best_score

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        skip: bool,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, skip, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()

class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        skip: bool,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, skip, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values

class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
