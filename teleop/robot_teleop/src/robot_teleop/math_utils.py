from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# PICO/OpenXR axes are x=right, y=up, z=back. The reference teleop stack applies
# this basis change before IK: robot xyz = [-pico_z, -pico_x, pico_y].
R_PICO_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def pose7_to_pos_quat_xyzw(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an XRoboToolkit pose [x, y, z, qx, qy, qz, qw] into position and quaternion."""
    pose = np.asarray(pose, dtype=float)
    pos = pose[:3].copy()
    quat = pose[3:7].copy()
    n = np.linalg.norm(quat)
    if n < 1e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat /= n
    return pos, quat


def transform_pico_pose_to_robot(
    pos_pico: np.ndarray,
    quat_pico_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a PICO/OpenXR pose into the robot world frame."""
    pos = R_PICO_TO_ROBOT @ np.asarray(pos_pico, dtype=float)
    rot_pico = Rotation.from_quat(quat_pico_xyzw).as_matrix()
    rot_robot = R_PICO_TO_ROBOT @ rot_pico @ R_PICO_TO_ROBOT.T
    quat = Rotation.from_matrix(rot_robot).as_quat()
    quat /= np.linalg.norm(quat)
    return pos, quat


def relative_pose(
    ref_pos: np.ndarray,
    ref_quat_xyzw: np.ndarray,
    cur_pos: np.ndarray,
    cur_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose of `cur` expressed in the `ref` frame."""
    ref_r = Rotation.from_quat(ref_quat_xyzw)
    cur_r = Rotation.from_quat(cur_quat_xyzw)
    rel_pos = ref_r.inv().apply(cur_pos - ref_pos)
    rel_quat = (ref_r.inv() * cur_r).as_quat()
    return rel_pos, rel_quat


def apply_relative_pose(
    base_pos: np.ndarray,
    base_quat_xyzw: np.ndarray,
    rel_pos: np.ndarray,
    rel_quat_xyzw: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a relative pose on top of `base`, scaling the translation part."""
    base_r = Rotation.from_quat(base_quat_xyzw)
    target_pos = base_pos + base_r.apply(rel_pos * scale)
    target_quat = (base_r * Rotation.from_quat(rel_quat_xyzw)).as_quat()
    return target_pos, target_quat


def clamp_to_limits(values: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> np.ndarray:
    """Clamp values to [lowers, uppers], ignoring non-finite bounds."""
    out = values.copy()
    finite_lower = np.isfinite(lowers)
    finite_upper = np.isfinite(uppers)
    out[finite_lower] = np.maximum(out[finite_lower], lowers[finite_lower])
    out[finite_upper] = np.minimum(out[finite_upper], uppers[finite_upper])
    return out
