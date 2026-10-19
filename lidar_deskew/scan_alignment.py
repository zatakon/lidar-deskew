"""Motion of a lidar estimated from its own scans.

The motion within a scan has to be known to take it out of the points. Here it comes from the
scans themselves, no other sensor is needed:

  The trajectory of the lidar is a chain of poses, one at the middle of each scan, with a
  constant velocity between two neighbours. Each scan is aligned to a map of the scans before
  it by moving its pose; the velocity that unwarps the scan follows from the same pose, so the
  scan is unwarped while it is being aligned. The alignment minimizes the plane-to-plane (GICP)
  distance of a few thousand points sampled from the scan.

The scans are processed one at a time, and the map keeps only what the last scans saw, so
that long recordings can be processed.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from . import se3
from .voxelgrid import last_point_per_voxel, voxel_keys

log = logging.getLogger(__name__)

# The covariance of a point is modelled as a disc: this variance along the normal, 1 in the plane
_NORMAL_VARIANCE = 1e-3
# A scan with fewer usable points is not aligned, the lidar is taken to keep its motion
_MIN_POINTS = 100


@dataclass
class AlignmentOptions:
    min_distance: float = 3.0                 # [m] closer points are not used for the alignment (the car itself)
    target_num_points: int = 10000            # points sampled from a scan for the alignment
    sample_voxel: float = 0.5                 # [m] the sample is spread evenly over voxels of this size
    k_neighbors: int = 20                     # neighbours used to estimate the local plane of a point
    max_correspondence_distance: float = 1.0  # [m]
    map_min_point_distance: float = 0.05      # [m] spacing of the points kept in the alignment map
    # A map point is dropped when none of the last this many scans saw it again (0: never)
    map_window_scans: int = 30
    max_iterations: int = 30
    # How strongly a scan is expected to continue the motion of the previous one. Only there
    # to keep the directions the scene does not constrain (e.g. along a straight wall) in place.
    motion_prior_sigma_trans: float = 0.05    # [m]
    motion_prior_sigma_rot_deg: float = 1.0   # [deg]
    # The first scan has nothing to be aligned to. It is unwarped with the motion found for the second
    # one and the second one is aligned again, up to this many times - so that a recording (or a time
    # window) may start while the car is already moving.
    bootstrap_iterations: int = 3
    # Correspondences of the first alignment of the second scan may be this far apart [m]: the motion
    # is not known yet, at 10 m/s the lidar moves 1 m per scan
    bootstrap_correspondence_distance: float = 2.5
    seed: int = 0


@dataclass
class ScanMotion:
    """Pose of the lidar at the middle of a scan, in the frame of the middle of the first scan."""
    start: float             # [s] time of the first point
    duration: float          # [s] until the start of the next scan
    T_mid: np.ndarray
    num_points: int = 0
    num_correspondences: int = 0
    iterations: int = 0
    residual: float = 0.0    # RMS plane-to-plane distance of the aligned sample [m]
    aligned: bool = True     # False: too few points, the motion of the previous scan was continued


# ---------------------------------------------------------------------------------------
# Per-point times
# ---------------------------------------------------------------------------------------
def scan_durations(starts: np.ndarray) -> np.ndarray:
    """Time from the start of each scan to the start of the next one (the last one repeats)."""
    if starts.shape[0] < 2:
        return np.full(starts.shape[0], 0.1)
    periods = np.diff(starts)
    median = float(np.median(periods))
    # A dropped scan would make one period twice as long; the lidar still took one period for the scan
    periods = np.where(periods > 1.5 * median, median, periods)
    return np.append(periods, periods[-1])


_warned_about_times = False


def point_time_fractions(times: Optional[np.ndarray], count: int, duration: float) -> np.ndarray:
    """
    Time of every point as a fraction of the scan, 0 at its start and 1 at the start of the next scan.

    Without per-point times in the recording the points are assumed to be in the order they
    were measured in, evenly spread over the scan.
    """
    global _warned_about_times
    if times is not None and count > 1 and float(times.max()) < 0.1 * duration:
        if not _warned_about_times:
            log.warning(f'The per-point times span only {float(times.max()) * 1000:.1f} ms of a {duration * 1000:.0f} ms '
                        f'scan - ignored, the points are taken as evenly spread over the scan in their order')
            _warned_about_times = True
        times = None
    if times is None:
        return np.arange(count, dtype=np.float64) / max(count, 1)
    return np.clip(times / duration, 0.0, 1.0)


# ---------------------------------------------------------------------------------------
# Sampling and normals
# ---------------------------------------------------------------------------------------
def sample_evenly(points: np.ndarray, target_num_points: int, voxel: float, rng: np.random.Generator) -> np.ndarray:
    """
    Indices of up to about `target_num_points` points, the same number from each occupied voxel.

    A plain random sample would mostly take the dense ground next to the car; this one keeps
    the far, sparse structures that constrain the alignment best.
    """
    count = points.shape[0]
    if target_num_points >= 0.99 * count:
        return np.arange(count)

    keys = voxel_keys(points, voxel)
    priority = rng.random(count)
    order = np.lexsort((priority, keys))
    sorted_keys = keys[order]
    is_start = np.empty(count, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    rank_in_voxel = np.arange(count) - np.repeat(starts, np.diff(np.append(starts, count)))

    points_per_voxel = int(np.ceil(target_num_points / starts.shape[0]))
    indices = order[rank_in_voxel < points_per_voxel]

    max_num_points = int(target_num_points * 1.2)
    if indices.shape[0] > max_num_points:
        indices = rng.choice(indices, max_num_points, replace=False)
    return np.sort(indices)


def estimate_normals(points: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Unit normal of the plane through the given neighbours of each point."""
    neighborhood = points[neighbors]                         # (N, k, 3)
    centered = neighborhood - neighborhood.mean(axis=1, keepdims=True)
    covariances = np.einsum('nki,nkj->nij', centered, centered)
    _, eigenvectors = np.linalg.eigh(covariances)
    return np.ascontiguousarray(eigenvectors[:, :, 0])       # eigenvector of the smallest eigenvalue


# ---------------------------------------------------------------------------------------
# Motion within a scan
# ---------------------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _rotation(wx, wy, wz, out):
    """Rotation matrix of the rotation vector (wx, wy, wz) into `out` (3,3)."""
    theta2 = wx * wx + wy * wy + wz * wz
    if theta2 < 1e-16:
        a = 1.0
        b = 0.5
    else:
        theta = np.sqrt(theta2)
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta2
    out[0, 0] = 1.0 - b * (wy * wy + wz * wz)
    out[0, 1] = -a * wz + b * wx * wy
    out[0, 2] = a * wy + b * wx * wz
    out[1, 0] = a * wz + b * wx * wy
    out[1, 1] = 1.0 - b * (wx * wx + wz * wz)
    out[1, 2] = -a * wx + b * wy * wz
    out[2, 0] = -a * wy + b * wx * wz
    out[2, 1] = a * wx + b * wy * wz
    out[2, 2] = 1.0 - b * (wx * wx + wy * wy)


@njit(cache=True, inline='always')
def _right_jacobian(wx, wy, wz, out):
    """Right Jacobian of SO(3) at the rotation vector (wx, wy, wz) into `out` (3,3)."""
    theta2 = wx * wx + wy * wy + wz * wz
    if theta2 < 1e-16:
        a = 0.5
        b = 1.0 / 6.0
    else:
        theta = np.sqrt(theta2)
        a = (1.0 - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
    out[0, 0] = 1.0 - b * (wy * wy + wz * wz)
    out[0, 1] = a * wz + b * wx * wy
    out[0, 2] = -a * wy + b * wx * wz
    out[1, 0] = -a * wz + b * wx * wy
    out[1, 1] = 1.0 - b * (wx * wx + wz * wz)
    out[1, 2] = a * wx + b * wy * wz
    out[2, 0] = a * wy + b * wx * wz
    out[2, 1] = -a * wx + b * wy * wz
    out[2, 2] = 1.0 - b * (wx * wx + wy * wy)


@njit(cache=True, parallel=True)
def unwarp(points, fractions, R_anchor, t_anchor, translation, rotation):
    """
    Points of a scan moved into the map frame, each by the pose at its own time:
        R(f) = R_anchor * exp(f * rotation),   t(f) = t_anchor + f * translation
    """
    count = points.shape[0]
    out = np.empty((count, 3), dtype=np.float64)
    theta = np.sqrt(rotation[0] ** 2 + rotation[1] ** 2 + rotation[2] ** 2)
    if theta > 1e-12:
        ax, ay, az = rotation[0] / theta, rotation[1] / theta, rotation[2] / theta
    else:
        ax, ay, az = 1.0, 0.0, 0.0
    for i in prange(count):
        f = fractions[i]
        px = np.float64(points[i, 0])
        py = np.float64(points[i, 1])
        pz = np.float64(points[i, 2])
        # Rodrigues: p cos + (a x p) sin + a (a . p) (1 - cos), around the common axis by f * theta
        sin_angle = np.sin(f * theta)
        cos_angle = np.cos(f * theta)
        dot = (ax * px + ay * py + az * pz) * (1.0 - cos_angle)
        qx = px * cos_angle + (ay * pz - az * py) * sin_angle + ax * dot
        qy = py * cos_angle + (az * px - ax * pz) * sin_angle + ay * dot
        qz = pz * cos_angle + (ax * py - ay * px) * sin_angle + az * dot
        out[i, 0] = R_anchor[0, 0] * qx + R_anchor[0, 1] * qy + R_anchor[0, 2] * qz + t_anchor[0] + f * translation[0]
        out[i, 1] = R_anchor[1, 0] * qx + R_anchor[1, 1] * qy + R_anchor[1, 2] * qz + t_anchor[1] + f * translation[1]
        out[i, 2] = R_anchor[2, 0] * qx + R_anchor[2, 1] * qy + R_anchor[2, 2] * qz + t_anchor[2] + f * translation[2]
    return out


@njit(cache=True)
def _linearize(points, normals, fractions, target_points, target_normals, correspondences,
               R_anchor, t_anchor, translation, rotation, with_derivatives):
    """
    Sum of the plane-to-plane (GICP) distances of the corresponding points, and its
    Gauss-Newton approximation w.r.t. the motion [translation, rotation].
    """
    H = np.zeros((6, 6))
    b = np.zeros(6)
    cost = 0.0
    count = 0
    scale = 1.0 - _NORMAL_VARIANCE

    R_inc = np.empty((3, 3))
    R = np.empty((3, 3))
    Jr = np.empty((3, 3))
    A = np.empty((3, 3))
    M = np.empty((3, 3))
    J = np.zeros((3, 6))
    D = np.empty((3, 3))

    for i in range(points.shape[0]):
        j = correspondences[i]
        if j < 0:
            continue
        f = fractions[i]
        _rotation(f * rotation[0], f * rotation[1], f * rotation[2], R_inc)
        for r in range(3):
            for c in range(3):
                R[r, c] = R_anchor[r, 0] * R_inc[0, c] + R_anchor[r, 1] * R_inc[1, c] + R_anchor[r, 2] * R_inc[2, c]

        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        ex = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz + t_anchor[0] + f * translation[0] - target_points[j, 0]
        ey = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz + t_anchor[1] + f * translation[1] - target_points[j, 1]
        ez = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz + t_anchor[2] + f * translation[2] - target_points[j, 2]

        # Sum of the two covariances: 2 I - scale * (n_target n_target^T + n_source n_source^T)
        nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
        sx = R[0, 0] * nx + R[0, 1] * ny + R[0, 2] * nz
        sy = R[1, 0] * nx + R[1, 1] * ny + R[1, 2] * nz
        sz = R[2, 0] * nx + R[2, 1] * ny + R[2, 2] * nz
        tx, ty, tz = target_normals[j, 0], target_normals[j, 1], target_normals[j, 2]
        A[0, 0] = 2.0 - scale * (tx * tx + sx * sx)
        A[0, 1] = -scale * (tx * ty + sx * sy)
        A[0, 2] = -scale * (tx * tz + sx * sz)
        A[1, 1] = 2.0 - scale * (ty * ty + sy * sy)
        A[1, 2] = -scale * (ty * tz + sy * sz)
        A[2, 2] = 2.0 - scale * (tz * tz + sz * sz)
        # ... and its inverse (symmetric 3x3)
        c00 = A[1, 1] * A[2, 2] - A[1, 2] * A[1, 2]
        c01 = A[0, 2] * A[1, 2] - A[0, 1] * A[2, 2]
        c02 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
        determinant = A[0, 0] * c00 + A[0, 1] * c01 + A[0, 2] * c02
        M[0, 0] = c00 / determinant
        M[0, 1] = c01 / determinant
        M[0, 2] = c02 / determinant
        M[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[0, 2]) / determinant
        M[1, 2] = (A[0, 1] * A[0, 2] - A[0, 0] * A[1, 2]) / determinant
        M[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[0, 1]) / determinant
        M[1, 0] = M[0, 1]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]

        mx = M[0, 0] * ex + M[0, 1] * ey + M[0, 2] * ez
        my = M[1, 0] * ex + M[1, 1] * ey + M[1, 2] * ez
        mz = M[2, 0] * ex + M[2, 1] * ey + M[2, 2] * ez
        cost += 0.5 * (ex * mx + ey * my + ez * mz)
        count += 1

        if not with_derivatives:
            continue

        # d(error)/d(translation) = f I;  d(error)/d(rotation) = -R hat(p) Jr(f rotation) f
        _right_jacobian(f * rotation[0], f * rotation[1], f * rotation[2], Jr)
        # D = R * hat(p)
        for r in range(3):
            D[r, 0] = R[r, 1] * pz - R[r, 2] * py
            D[r, 1] = R[r, 2] * px - R[r, 0] * pz
            D[r, 2] = R[r, 0] * py - R[r, 1] * px
        for r in range(3):
            J[r, 0] = 0.0
            J[r, 1] = 0.0
            J[r, 2] = 0.0
            J[r, r] = f
            for c in range(3):
                J[r, 3 + c] = -f * (D[r, 0] * Jr[0, c] + D[r, 1] * Jr[1, c] + D[r, 2] * Jr[2, c])

        for c in range(6):
            jm0 = J[0, c] * M[0, 0] + J[1, c] * M[1, 0] + J[2, c] * M[2, 0]
            jm1 = J[0, c] * M[0, 1] + J[1, c] * M[1, 1] + J[2, c] * M[2, 1]
            jm2 = J[0, c] * M[0, 2] + J[1, c] * M[1, 2] + J[2, c] * M[2, 2]
            b[c] += J[0, c] * mx + J[1, c] * my + J[2, c] * mz
            for c2 in range(c, 6):
                H[c, c2] += jm0 * J[0, c2] + jm1 * J[1, c2] + jm2 * J[2, c2]

    for c in range(6):
        for c2 in range(c):
            H[c, c2] = H[c2, c]
    return H, b, cost, count



class AlignmentMap:
    """Points with normals that the scans are aligned to."""

    def __init__(self, min_point_distance: float, window_scans: int = 0):
        self.min_point_distance = min_point_distance
        self.window_scans = window_scans
        self.reset()

    def reset(self) -> None:
        self.points = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.last_seen = np.zeros(0, dtype=np.int64)   # index of the last scan that saw the point
        self.tree: Optional[cKDTree] = None

    def insert(self, points: np.ndarray, normals: np.ndarray, scan_index: int = 0) -> None:
        # One point per cell of the minimal spacing, and none next to a point the map already has
        keep = last_point_per_voxel(points, self.min_point_distance)
        points, normals = points[keep], normals[keep]
        if self.tree is not None:
            distances, indices = self.tree.query(points, distance_upper_bound=self.min_point_distance, workers=-1)
            is_new = ~np.isfinite(distances)
            self.last_seen[indices[~is_new]] = scan_index
            points, normals = points[is_new], normals[is_new]
        self.points = np.concatenate([self.points, points])
        self.normals = np.concatenate([self.normals, normals])
        self.last_seen = np.concatenate([self.last_seen, np.full(points.shape[0], scan_index, dtype=np.int64)])
        if self.window_scans > 0:
            recent = self.last_seen > scan_index - self.window_scans
            if not recent.all():
                self.points, self.normals, self.last_seen = self.points[recent], self.normals[recent], self.last_seen[recent]
        self.tree = cKDTree(self.points)

    def nearest(self, points: np.ndarray, max_distance: float) -> np.ndarray:
        """Index of the nearest map point of each point, -1 where there is none within `max_distance`."""
        distances, indices = self.tree.query(points, distance_upper_bound=max_distance, workers=-1)
        return np.where(np.isfinite(distances), indices, -1).astype(np.int64)


def align_scan(points: np.ndarray, normals: np.ndarray, fractions: np.ndarray, alignment_map: AlignmentMap,
               T_anchor: np.ndarray, predicted_motion: np.ndarray, options: AlignmentOptions,
               use_prior: bool = True, max_correspondence_distance: Optional[float] = None):
    """
    Motion [translation (map frame), rotation vector] of the lidar from `T_anchor` that aligns a scan with the map.

    A point with the fraction f is moved by the pose `T_anchor` advanced by f times the motion.

    @param predicted_motion: the motion to start from and to stay close to where the scene does not say otherwise
    @param use_prior: False where there is no motion to expect yet (the prediction is only where to start)
    @return: motion, number of correspondences, iterations, RMS residual [m]
    """
    R_anchor = np.ascontiguousarray(T_anchor[:3, :3])
    t_anchor = np.ascontiguousarray(T_anchor[:3, 3])
    motion = predicted_motion.copy()
    max_distance = max_correspondence_distance or options.max_correspondence_distance

    prior_information = np.diag(np.concatenate([
        np.full(3, 1.0 / options.motion_prior_sigma_trans ** 2),
        np.full(3, 1.0 / np.radians(options.motion_prior_sigma_rot_deg) ** 2),
    ])) * (1.0 if use_prior else 0.0)

    def prior_cost(current):
        difference = current - predicted_motion
        return 0.5 * difference @ prior_information @ difference

    num_correspondences = 0
    iterations = 0
    cost = 0.0
    for iterations in range(1, options.max_iterations + 1):
        transformed = unwarp(points, fractions, R_anchor, t_anchor, motion[:3], motion[3:])
        correspondences = alignment_map.nearest(transformed, max_distance)

        H, b, cost, num_correspondences = _linearize(
            points, normals, fractions, alignment_map.points, alignment_map.normals, correspondences,
            R_anchor, t_anchor, motion[:3], motion[3:], True)
        H += prior_information
        b += prior_information @ (motion - predicted_motion)
        total_cost = cost + prior_cost(motion)

        try:
            step = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break  # nothing constrains the motion (no correspondences and no prior)
        # Accept the step only if it lowers the cost (with the same correspondences), else shorten it
        for _ in range(6):
            candidate = motion + step
            _, _, candidate_cost, _ = _linearize(
                points, normals, fractions, alignment_map.points, alignment_map.normals, correspondences,
                R_anchor, t_anchor, candidate[:3], candidate[3:], False)
            if candidate_cost + prior_cost(candidate) <= total_cost:
                break
            step = 0.5 * step
        else:
            break
        motion = candidate

        if np.linalg.norm(step[:3]) < 1e-5 and np.linalg.norm(step[3:]) < 1e-6:
            break

    # Plane-to-plane RMS: the weights are ~1/(2 * normal variance) along the normal
    residual = float(np.sqrt(2.0 * cost / max(num_correspondences, 1) * 2.0 * _NORMAL_VARIANCE))
    return motion, num_correspondences, iterations, residual


def _advance(T: np.ndarray, motion: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """The pose `T` advanced by `fraction` times the motion [translation (map frame), rotation vector]."""
    advanced = np.eye(4)
    advanced[:3, :3] = T[:3, :3] @ se3.exp_so3(fraction * motion[3:])
    advanced[:3, 3] = T[:3, 3] + fraction * motion[:3]
    return advanced


def _motion_between(T_from: np.ndarray, T_to: np.ndarray) -> np.ndarray:
    return np.concatenate([T_to[:3, 3] - T_from[:3, 3], se3.log_so3(T_from[:3, :3].T @ T_to[:3, :3])])


class MotionEstimator:
    """
    Aligns the scans one after another and gives the pose of the lidar at the middle of each of them.

    Between the middle of the previous scan and the middle of the current one the lidar moves at a
    constant velocity, and keeps it until the end of the current scan. The alignment finds that
    motion; the points of the scan are at 0.5 .. 1.5 of it, as the scan starts half a scan after
    the middle of the previous one.

    The first scan defines the map frame (its middle is the identity). It has nothing to be aligned
    to, so it moves as the second scan says (see `_bootstrap`): the recording does not have to start
    with the car standing still.
    """

    def __init__(self, options: AlignmentOptions = AlignmentOptions()):
        self.options = options
        self.rng = np.random.default_rng(options.seed)
        self.map = AlignmentMap(options.map_min_point_distance, options.map_window_scans)
        self.motions: List[ScanMotion] = []
        self.T_previous = np.eye(4)
        self.previous_motion_body = np.zeros(6)  # translation in the frame of the previous pose, rotation vector
        self.first_scan = None

    def add(self, points: np.ndarray, fractions: np.ndarray, start: float, duration: float) -> ScanMotion:
        """
        @param points: (N,3) the valid points of the scan (finite), lidar frame
        @param fractions: (N,) their times as fractions of the scan (see `point_time_fractions`)
        """
        options = self.options
        index = len(self.motions)
        fractions = fractions + 0.5
        far_enough = np.linalg.norm(points, axis=1) >= options.min_distance
        points = np.ascontiguousarray(points[far_enough], dtype=np.float64)
        fractions = np.ascontiguousarray(fractions[far_enough])
        predicted = np.concatenate([self.T_previous[:3, :3] @ self.previous_motion_body[:3],
                                    self.previous_motion_body[3:]])

        if points.shape[0] < _MIN_POINTS:
            motion, scan = predicted, None
            num_correspondences, iterations, residual = 0, 0, 0.0
            log.warning(f'scan {index + 1}: only {points.shape[0]} points beyond {options.min_distance} m - not aligned, '
                        f'the motion of the previous scan is continued')
        else:
            sample = sample_evenly(points, options.target_num_points, options.sample_voxel, self.rng)
            sample_points = points[sample]
            sample_fractions = fractions[sample]
            k = min(options.k_neighbors, sample_points.shape[0])
            _, neighbors = cKDTree(sample_points).query(sample_points, k=k, workers=-1)
            normals = estimate_normals(sample_points, neighbors)
            scan = (sample_points, sample_fractions, neighbors)

            if self.map.tree is None:
                # The first usable scan: nothing to align it to yet (all scans before it, if any, had no
                # points, so the lidar is still in the identity)
                motion = predicted
                num_correspondences, iterations, residual = 0, 0, 0.0
                self.first_scan = scan
            elif self.first_scan is not None and options.bootstrap_iterations > 0:
                motion, num_correspondences, iterations, residual = self._bootstrap(
                    sample_points, normals, sample_fractions)
            else:
                motion, num_correspondences, iterations, residual = align_scan(
                    sample_points, normals, sample_fractions, self.map, self.T_previous, predicted, options)

        if scan is not None:
            # The unwarped sample goes into the map, with the normals of the unwarped points
            sample_points, sample_fractions, neighbors = scan
            unwarped = unwarp(sample_points, sample_fractions, np.ascontiguousarray(self.T_previous[:3, :3]),
                              np.ascontiguousarray(self.T_previous[:3, 3]), motion[:3], motion[3:])
            self.map.insert(unwarped, estimate_normals(unwarped, neighbors), index)

        T_mid = _advance(self.T_previous, motion)
        result = ScanMotion(start=float(start), duration=float(duration), T_mid=T_mid, num_points=int(points.shape[0]),
                            num_correspondences=int(num_correspondences), iterations=int(iterations),
                            residual=residual, aligned=scan is not None)
        self.motions.append(result)
        log.debug(f'scan {index + 1:5d}: moved {np.linalg.norm(motion[:3]) * 1000:6.1f} mm, '
                  f'{np.degrees(np.linalg.norm(motion[3:])):5.2f} deg, {num_correspondences:5d} correspondences, '
                  f'residual {residual * 1000:5.1f} mm, {iterations:2d} iterations, map {self.map.points.shape[0]:7d} points')

        self.previous_motion_body = np.concatenate([self.T_previous[:3, :3].T @ motion[:3], motion[3:]])
        self.T_previous = T_mid
        return result

    def _bootstrap(self, points: np.ndarray, normals: np.ndarray, fractions: np.ndarray):
        """
        Motion from the middle of the first scan to the middle of the second one.

        The map holds only the first scan, taken as not moving. When the car already moves, that scan is
        smeared by the distance driven during it (30 cm at 3 m/s) and the motion to expect is not known.
        So the second scan is aligned without a motion prior and with far apart correspondences first;
        then the first scan is unwarped with the motion found (the same constant velocity through both
        scans), the map is made of it again and the second scan is aligned again. The map ends up
        holding the unwarped first scan.
        """
        options = self.options
        first_points, first_fractions, first_neighbors = self.first_scan
        identity, zero = np.eye(3), np.zeros(3)
        motion, num_correspondences, iterations, residual = align_scan(
            points, normals, fractions, self.map, np.eye(4), np.zeros(6), options, use_prior=False,
            max_correspondence_distance=max(options.bootstrap_correspondence_distance, options.max_correspondence_distance))
        for _ in range(options.bootstrap_iterations):
            # The points of the first scan lie at -0.5 .. 0.5 of the motion from its own middle
            unwarped = unwarp(first_points, first_fractions - 1.0, identity, zero, motion[:3], motion[3:])
            self.map.reset()
            self.map.insert(unwarped, estimate_normals(unwarped, first_neighbors), 0)
            previous = motion
            motion, num_correspondences, iterations, residual = align_scan(
                points, normals, fractions, self.map, np.eye(4), previous, options, use_prior=False)
            if np.linalg.norm(motion[:3] - previous[:3]) < 1e-4 and np.linalg.norm(motion[3:] - previous[3:]) < 1e-5:
                break
        self.first_scan = None
        return motion, num_correspondences, iterations, residual


# ---------------------------------------------------------------------------------------
# The estimated trajectory
# ---------------------------------------------------------------------------------------
def _segments(motions: List[ScanMotion], index: int):
    """
    The two pieces of the trajectory a scan lies on, each as (anchor pose, motion, fraction offset):
    its first half between the middle of the previous scan and its own middle, the second half
    between its own middle and the middle of the next scan. At the ends of the recording the
    neighbouring piece is continued.
    """
    count = len(motions)
    if count == 1:
        still = (motions[0].T_mid, np.zeros(6), 0.0)
        return still, still

    def piece(first: int):
        return motions[first].T_mid, _motion_between(motions[first].T_mid, motions[first + 1].T_mid)

    if index == 0:
        anchor, motion = piece(0)
        return (anchor, motion, -0.5), (anchor, motion, -0.5)
    before = (*piece(index - 1), 0.5)
    if index == count - 1:
        return before, before
    return before, (*piece(index), -0.5)


class LidarTrajectory:
    """The motion estimated from the scans, piecewise linear between the middles of the scans."""

    def __init__(self, motions: List[ScanMotion]):
        if not motions:
            raise ValueError('No scans')
        self.motions = motions
        self._starts = np.array([motion.start for motion in motions])

    def pose_at(self, stamp: float) -> np.ndarray:
        """Pose of the lidar at the given time, in the frame of the middle of the first scan."""
        index = int(np.clip(np.searchsorted(self._starts, stamp, side='right') - 1, 0, len(self.motions) - 1))
        fraction = (stamp - self.motions[index].start) / self.motions[index].duration
        first_half, second_half = _segments(self.motions, index)
        anchor, motion, offset = first_half if fraction < 0.5 else second_half
        return _advance(anchor, motion, fraction + offset)

    def deskew(self, index: int, points: np.ndarray, fractions: np.ndarray, reference_time: float) -> np.ndarray:
        """
        The points of scan `index`, each moved by the pose at its own time into the lidar frame at `reference_time`.

        @param fractions: times of the points as fractions of the scan (see `point_time_fractions`)
        """
        T_reference_inverse = se3.inverse(self.pose_at(reference_time))
        out = np.empty((points.shape[0], 3))
        first_half = fractions < 0.5
        for selection, (anchor, motion, offset) in zip((first_half, ~first_half), _segments(self.motions, index)):
            anchor = T_reference_inverse @ anchor
            out[selection] = unwarp(np.ascontiguousarray(points[selection]), fractions[selection] + offset,
                                    np.ascontiguousarray(anchor[:3, :3]), np.ascontiguousarray(anchor[:3, 3]),
                                    T_reference_inverse[:3, :3] @ motion[:3], motion[3:])
        return out
