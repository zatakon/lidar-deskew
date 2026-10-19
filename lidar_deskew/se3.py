"""Rigid transforms.

Conventions used in the whole package:
  - a pose is a 4x4 homogeneous matrix `T_a_b`, which maps points from the frame `b` to the frame `a`
  - the 7 values of a pose are `[tx, ty, tz, qx, qy, qz, qw]`
  - a tangent vector is `[vx, vy, vz, wx, wy, wz]` (translation first) and is applied
    on the right: `T * exp(xi)`
"""
import numpy as np


def quat_to_rot(q) -> np.ndarray:
    """Rotation matrix from the quaternion `[x, y, z, w]` (normalized first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rot_to_quat(R) -> np.ndarray:
    """Quaternion `[x, y, z, w]` (with w >= 0) from a rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def pose_from_values(values) -> np.ndarray:
    """4x4 pose from `[tx, ty, tz, qx, qy, qz, qw]`."""
    values = np.asarray(values, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(values[3:7])
    T[:3, 3] = values[0:3]
    return T


def pose_to_values(T) -> list:
    """`[tx, ty, tz, qx, qy, qz, qw]` of a 4x4 pose."""
    T = np.asarray(T, dtype=np.float64)
    return [*T[:3, 3].tolist(), *rot_to_quat(T[:3, :3]).tolist()]


def inverse(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    inv = np.eye(4)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv


def hat(w) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    W = hat(w)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(theta) / theta * W + (1.0 - np.cos(theta)) / theta ** 2 * W @ W


def log_so3(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-6:
        # Close to 180 deg the antisymmetric part vanishes, take the axis from the symmetric one
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        axis = A[:, k] / axis[k]
        if axis @ vee < 0.0:
            axis = -axis
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * vee


def _left_jacobian_so3(w) -> np.ndarray:
    theta = np.linalg.norm(w)
    W = hat(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * W + W @ W / 6.0
    return (np.eye(3) + (1.0 - np.cos(theta)) / theta ** 2 * W
            + (theta - np.sin(theta)) / theta ** 3 * W @ W)


def exp_se3(xi) -> np.ndarray:
    """Exponential map of `[v, w]` (the same as Sophus::SE3d::exp / gtsam with swapped halves)."""
    xi = np.asarray(xi, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = exp_so3(xi[3:6])
    T[:3, 3] = _left_jacobian_so3(xi[3:6]) @ xi[0:3]
    return T


def log_se3(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    w = log_so3(T[:3, :3])
    v = np.linalg.solve(_left_jacobian_so3(w), T[:3, 3])
    return np.concatenate([v, w])


def retract(T, xi) -> np.ndarray:
    """`T * exp(xi)`, re-orthonormalized so that the rotation does not drift over many updates."""
    T_new = np.asarray(T, dtype=np.float64) @ exp_se3(xi)
    T_new[:3, :3] = quat_to_rot(rot_to_quat(T_new[:3, :3]))
    return T_new


def pose_delta(T_a, T_b):
    """Translation [m] and rotation [rad] of `inv(T_a) * T_b`."""
    delta = inverse(T_a) @ np.asarray(T_b, dtype=np.float64)
    return float(np.linalg.norm(delta[:3, 3])), float(np.linalg.norm(log_so3(delta[:3, :3])))


def interpolate(T_a, T_b, t: float) -> np.ndarray:
    """Pose between `T_a` (t=0) and `T_b` (t=1): rotation and translation interpolated independently."""
    T = np.eye(4)
    T[:3, :3] = T_a[:3, :3] @ exp_so3(t * log_so3(T_a[:3, :3].T @ T_b[:3, :3]))
    T[:3, 3] = (1.0 - t) * T_a[:3, 3] + t * T_b[:3, 3]
    return T


def rpy_from_rot(R):
    """Roll, pitch, yaw (URDF convention, R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix of roll, pitch, yaw [rad] (URDF convention, R = Rz(yaw) Ry(pitch) Rx(roll))."""
    return exp_so3([0.0, 0.0, yaw]) @ exp_so3([0.0, pitch, 0.0]) @ exp_so3([roll, 0.0, 0.0])
