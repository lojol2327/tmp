# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta, within_fov_cone
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

import math
import json
import base64
from datetime import datetime
from scipy.ndimage import rotate
from openai import OpenAI
client = OpenAI()

PROMPT_SEPARATOR = "|"

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "14181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "14184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "14183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "14185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._done_reinitializing = True # Convert to False when reinitializing is required
        self._is_looking_around = False # Set mode as looking around
        self._looking_around_steps = 0 
        self._temp_goal = np.zeros(2) # temporary goal selected from descriptions
        self._num_rotates = 0
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        self._images = [] # temporary image store for panorama
        self._observations_list = [] # temporary yaw store
        self._descriptions = [] # description extracted by OpenAI
        self._rooms = [] # sequence of rooms
        self._last_initialized_pos = np.zeros(2)
        self._last_initialized_step = 0
        self._detected_verified = False
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._done_reinitializing = True
        self._is_looking_around = False
        self._looking_around_steps = 0
        self._temp_goal = np.zeros(2)
        self._num_rotates = 0
        self._images = []
        self._observations_list = []
        self._descriptions = []
        self._rooms = []
        self._last_initialized_pos = np.zeros(2)
        self._last_initialized_step = 0
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self._detected_verified = False

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        skip: bool,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        robot_heading = self._observations_cache["robot_heading"]
        goal = self._get_target_object_location(robot_xy)
        
        frontiers = self._observations_cache["frontier_sensor"]
        # Right after (re)initialize, operate new value map
        if len(self._images) == 12 and len(frontiers) > 0:
            if self._num_steps > 12 or (self._num_steps <= 12 and self._is_reinitialize_required(observations)):
                self._descriptions, direction = self._extract_descriptions()
                # Select new best frontier
                best_frontier, best_value = self._select_frontier_from_description(frontiers, direction)
                print(f"*NEW* Best frontier : {best_frontier}, Best value: {best_value*100:.2f}%") # for simple test
                self._is_looking_around = True
                self._temp_goal = best_frontier
            else: # iniialization is completed but reinitializing not required
                panorama = self._panorama()
                self._describe_room(panorama)
            self._images.clear()
            self._observations_list.clear()
            
        # If the robot reaches to the temp goal, finish the step
        if self._is_looking_around:
            if ((robot_xy[0] - self._temp_goal[0])*(robot_xy[0] - self._temp_goal[0]) + (robot_xy[1] - self._temp_goal[1])*(robot_xy[1] - self._temp_goal[1]) < 0.25) or len(frontiers) <= 1 or (not self._temp_goal in frontiers and within_fov_cone(robot_xy, robot_heading, np.radians(90), 10.0, frontiers).size==0) or self._looking_around_steps > 30: 
                self._is_looking_around = False
                self._looking_around_steps = 0
                self._temp_goal = np.zeros(2)
                print("Looking around is finished")

        

        """
        """
        

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            if not self._done_reinitializing:
                mode = "reinitialize"
                pointnav_action = self._reinitialize()
            else:
                if ((not self._is_looking_around) or (not self._temp_goal in frontiers)) and self._is_reinitialize_required(observations):
                #if (not self._is_looking_around) and self._is_reinitialize_required(observations): # no time
                    if self._is_looking_around:
                        self._is_looking_around = False
                        self._looking_around_steps = 0
                        self._temp_goal = np.zeros(2)
                        print("Looking around is finished")
                    self._done_reinitializing = False
                    self._last_initialized_pos = self._observations_cache["robot_xy"]
                    self._last_initialized_step = self._num_steps
                    mode = "reinitialize"
                    pointnav_action = self._reinitialize()
                elif self._is_looking_around:
                    mode = "looking around"
                    self._looking_around_steps+=1
                    pointnav_action = self._pointnav(self._temp_goal[:2], stop=False)
                else:
                    # Explore 대신 가장 angular closest frontier로로
                    # mode = "explore w/o value map"
                    # if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                    #     print("No frontiers found during exploration, stopping.")
                    #     pointnav_action = self._stop_action
                    # else: 
                    #     directions = frontiers - robot_xy
                    #     angles = np.arctan2(directions[:, 1], directions[:, 0])
                    #     angle_diffs = np.abs(np.mod(angles - robot_heading + np.pi, 2 * np.pi) - np.pi)
                    #     selected_frontier = frontiers[np.argmin(angle_diffs)]
                    #     print("Angular closest frontier selected.")
                    #     print(f"Angle: {np.min(angle_diffs)*180/np.pi:.2f}, Distance: {np.linalg.norm(directions[np.argmin(angle_diffs)])}")
                    #     pointnav_action = self._explore(selected_frontier)
                    mode = "explore"
                    pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)
        
        if skip:
            print("skip")
            pointnav_action = self._stop_action

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy} | Position: {robot_xy} | Heading: {int(robot_heading * 180 / math.pi)}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states
    
    def _panorama(self) -> str:
        if len(self._images) == 12:
            cropped_images = []
            for i, image in enumerate(self._images):
                # Check image dimensions
                if image.shape[1] != 640 or image.shape[0] != 480:
                    raise ValueError(f"Image {i+1} has invalid dimensions {image.shape}. Expected 640x480.")

                # Crop the center part (210x480)
                x_center = image.shape[1] // 2
                x_start = x_center - 210 // 2
                x_end = x_center + 210 // 2
                cropped_image = image[:, x_start:x_end]  # Crop center horizontally
                cropped_images.append(cropped_image)

            # Concatenate the cropped images horizontally to create a panorama
            center_index = 6
            reordered_images = cropped_images[center_index:] + cropped_images[:center_index]
            reordered_images.reverse()
            panorama = np.hstack(reordered_images)
            
            def encode_image_from_array(image: np.ndarray) -> str:
                # Encode the image to a JPEG format in memory
                success, encoded_image = cv2.imencode(".jpg", image)
                if not success:
                    raise ValueError("Image encoding failed")

                # Convert the encoded image to a base64 string
                base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
                return base64_image
            
            base64_image = encode_image_from_array(panorama)
            
            return base64_image
        
    def _describe_room(self, image):
        room_prompt = "The given 360 degree image is the robot's current observation. Explain about the room the robot is currently located. Answer in ONE sentece without verb. "
        if not len(self._rooms) == 0:
            last_three_rooms = self._rooms[-3:]
            formatted_rooms = " -> ".join(f"({room})" for room in last_three_rooms)
            room_prompt = room_prompt + "The sequence of the rooms you already visited are as follows: " + formatted_rooms + " If the current room seems identical to one of the visited rooms, copy it. Otherwise, describe about the new room."
        room_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {
                #     "role": "system",
                #     "content": "You are a helpful robot agent designed to walk around an indoor environment like a house. Keep in mind that you can't go inside the closed door, and you can't take the stairs. The specific target object will be given for you, and you have to navigate toward the target with the shortest and most efficient path."
                # },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": room_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "low"
                            }
                        },
                    ],
                }
            ],
            temperature=0
        )
        #log
        print(room_completion.usage)
        room_answer = room_completion.choices[0].message
        if (room_answer.refusal):
            print(room_answer.refusal)
        else: 
            room_new = room_answer.content
            if not self._rooms or self._rooms[-1] != room_new:
                self._rooms.append(room_new)
            formatted_rooms = " -> ".join(f"({room})" for room in self._rooms)
            print(f"Visited Rooms: {formatted_rooms}")
        
    def _extract_descriptions(self) -> tuple[list[str], int]:
        """
        Extract descriptions for selecting the most promising frontier 
        based on the circular observation
        """
        if len(self._images) == 12:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
            base64_image = self._panorama()
                
            # OPENAI called
            print(f"***** OpenAI API called ({timestamp}) *****")
            prompt1 = "1. The given 360 degree image is the current observation. Based on the image, estimate where the most promising direction is to find a target_object. If the target is not directly found in current observation, it would be necessary to lead the robot to explore unobserved regions. "
            prompt1 = prompt1 + "Note that the middle part of the image is the direction the robot is currently looking at, and both sides are areas that have already been explored behind the robot. "
            prompt1 = prompt1 + "How many pixels away from the left end of the image is the most promising direction? Just answer as a number and format the answer as json with a key \"direction description\". "
            if not len(self._rooms) == 0:
                last_three_rooms = self._rooms[-3:]
                formatted_room = " -> ".join(f"({room})" for room in last_three_rooms)
                prompt1 = prompt1 + "The sequence of the rooms you already visited are as follows: " + formatted_room + " Refer to the information that you didn't find the target in the visited rooms. "
            prompt2 = "2. Describe about the specific point on that direction to lead the robot at that point. Answer each description as one sentence without verbs, and format them as json with a single key \"target descriptions\"."
            prompt = prompt1 + prompt2

            target = self._target_object.split(PROMPT_SEPARATOR)[0]
            prompt = prompt.replace('target_object', target)
            print("Q: ", prompt)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful robot agent designed to walk around an indoor environment like a house. Keep in mind that you can't go inside the closed door, and you can't take the stairs. The specific target object will be given for you, and you have to navigate toward the target with the shortest and most efficient path."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "descriptions_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                            "direction_description": {
                                "type": "integer",
                                "description": "Pixel information of a promising direction."
                            },
                            "target_description": {
                                "type": "array",
                                "description": "A list of descriptions",
                                "items": {
                                    "type": "string",
                                    "description": "A description of a target point"
                                }
                            }
                            },
                            "required": [
                                "direction_description",
                                "target_description"
                            ],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            # log
            print(completion.usage)
            
            answer = completion.choices[0].message
            try:
                content_dict = json.loads(answer.content)
                direction = content_dict.get("direction_description", 1000)
                if not isinstance(direction, int):
                    try:
                        direction = int(direction)  # Attempt to convert it to an integer
                    except (ValueError, TypeError):
                        direction = 1000  # Fallback value if conversion fails
                descriptions = content_dict.get("target_description", [])
                if not isinstance(descriptions, list):
                    descriptions = [str(descriptions)]  # Convert a single string to a list
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}. Raw content: {answer.content}")
                # Provide a default value or handle it gracefully
                direction = 1000
                descriptions = ["It seems like " + self._target_object.split(PROMPT_SEPARATOR)[0] + " ahead."]        
            
            self._describe_room(base64_image) # only for v1
            
            if (answer.refusal):
                print(answer.refusal)
                direction = 1000
                descriptions = ["It seems like " + self._target_object.split(PROMPT_SEPARATOR)[0] + " ahead."]        
                return descriptions, direction
            else:
                print(f"Direction: {direction}px")
                print(f"Generated text prompt: {descriptions}")
                return descriptions, direction
        else:
            print("Failed to extract descriptions due to insufficient images.")
            return None   

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue   
            
            """
            # verification
            # Further, adjust detection threshold
            x_center = rgb.shape[1] // 2
            x_start = x_center - 512 // 2
            x_end = x_center + 512 // 2
            cropped_rgb = rgb[:, x_start:x_end]  # Crop center horizontally
            
            def encode_image_from_array(image: np.ndarray) -> str:
                # Encode the image to a JPEG format in memory
                success, encoded_image = cv2.imencode(".jpg", image)
                if not success:
                    raise ValueError("Image encoding failed")

                # Convert the encoded image to a base64 string
                base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
                return base64_image
            
            if not self._detected_verified:
                base64_image = encode_image_from_array(cropped_rgb)
                
                print("*******OpenAI called*******")
                target = detections.phrases[idx]
                prompt = f"\"{target}\" is detected in the given image. Verify whether the target is really in the image based on the spatial context. Return True if the target exists, and return False if not."
                print(prompt)
                ver_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                            ],
                        }
                    ],
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "verification_schema",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "verification": {
                                        "type": "boolean",
                                        "description": "Verify whether target is in the image."
                                    }
                                },
                                "required": [
                                    "verification"
                                ],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                # log
                print(ver_completion.usage)
                
                answer = ver_completion.choices[0].message
                content_dict = json.loads(answer.content)
                verification = content_dict.get("verification")
                if not verification:
                    print("Target is detected but not verified.")
                    continue
                else:
                    self._detected_verified = True
            """

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
