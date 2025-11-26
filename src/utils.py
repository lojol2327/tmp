import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
from typing import Optional

def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def get_pts_angle_aeqa(init_pts, init_quat):
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    all_distances = []
    for viewpoint in viewpoints:
        path = habitat_sim.ShortestPath()
        path.requested_start = curr_pts
        path.requested_end = viewpoint
        found_path = pathfinder.find_path(path)
        if not found_path:
            all_distances.append(np.inf)
        else:
            all_distances.append(path.geodesic_distance)
    return min(all_distances)


def get_point_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    cam_intr: np.ndarray,
    cam_pose: np.ndarray,
    use_mask_center: bool = False
) -> Optional[np.ndarray]:
    """
    마스크와 깊이 정보로부터 3D 월드 좌표를 계산합니다.
    """
    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask)
    
    if use_mask_center:
        center_x, center_y = int(xs.mean()), int(ys.mean())
    else:
        # 깊이 값이 유효한 픽셀들만 필터링
        valid_depth_mask = (depth[ys, xs] > 0.1) & (depth[ys, xs] < 5.0)
        if not np.any(valid_depth_mask):
            return None # 유효 깊이 없음
        
        ys, xs = ys[valid_depth_mask], xs[valid_depth_mask]
        
        # 유효 깊이 픽셀들의 중앙점 사용
        center_x, center_y = int(xs.mean()), int(ys.mean())

    point_depth = depth[center_y, center_x]
    if np.isnan(point_depth) or point_depth <= 0.1:
        return None

    # 2D pixel to 3D point in camera frame
    cam_fx, cam_fy = cam_intr[0, 0], cam_intr[1, 1]
    cam_cx, cam_cy = cam_intr[0, 2], cam_intr[1, 2]
    
    x_cam = (center_x - cam_cx) * point_depth / cam_fx
    y_cam = (center_y - cam_cy) * point_depth / cam_fy
    z_cam = point_depth
    
    # Camera frame (-Z forward) to Habitat frame (-Z forward, Y up)
    # The coordinate systems are aligned, but Habitat uses homogeneous coordinates.
    p_cam_h = np.array([x_cam, y_cam, z_cam, 1.0])

    # Transform to world coordinate system
    p_world = (cam_pose @ p_cam_h)[:3]
    
    return p_world
