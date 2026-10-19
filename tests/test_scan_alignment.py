import numpy as np
import pytest

from helpers import SCAN_PERIOD, ROOM_HALF, accelerating, distance_to_walls, moving, pose_matrix, scans
from lidar_deskew import se3
from lidar_deskew.scan_alignment import (AlignmentMap, AlignmentOptions, LidarTrajectory, MotionEstimator, _linearize,
                                         point_time_fractions, sample_evenly, scan_durations, unwarp)


def estimate(scan_list, options=AlignmentOptions(min_distance=0.5)):
    estimator = MotionEstimator(options)
    for stamp, points, times in scan_list:
        estimator.add(points, point_time_fractions(times, points.shape[0], SCAN_PERIOD), stamp, SCAN_PERIOD)
    return LidarTrajectory(estimator.motions)


def test_unwarp_matches_per_point_poses():
    rng = np.random.default_rng(0)
    points = rng.normal(scale=10.0, size=(200, 3))
    fractions = rng.uniform(size=200)
    T_begin = se3.exp_se3([1.0, -2.0, 0.5, 0.1, -0.2, 0.3])
    translation = np.array([0.3, -0.1, 0.05])
    rotation = np.array([0.02, -0.03, 0.05])

    result = unwarp(points, fractions, np.ascontiguousarray(T_begin[:3, :3]), np.ascontiguousarray(T_begin[:3, 3]),
                    translation, rotation)
    for point, fraction, expected in zip(points, fractions, result):
        R = T_begin[:3, :3] @ se3.exp_so3(fraction * rotation)
        assert np.allclose(R @ point + T_begin[:3, 3] + fraction * translation, expected, atol=1e-10)


def test_alignment_derivatives():
    """Gradient of the alignment cost against finite differences."""
    rng = np.random.default_rng(1)
    count = 300
    points = rng.normal(scale=8.0, size=(count, 3))
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    fractions = rng.uniform(size=count)
    correspondences = np.arange(count)
    R_begin = np.ascontiguousarray(se3.exp_so3([0.05, -0.02, 0.1]))
    t_begin = np.array([0.2, 0.1, -0.05])
    motion = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03])

    # Targets close to the unwarped points: the residuals are small, as they are near the solution
    target_points = unwarp(points, fractions, R_begin, t_begin, motion[:3], motion[3:]) \
        + rng.normal(scale=0.002, size=(count, 3))
    target_normals = np.array([R_begin @ se3.exp_so3(f * motion[3:]) @ n for f, n in zip(fractions, normals)])
    target_normals += rng.normal(scale=0.05, size=(count, 3))
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)

    def cost(x):
        return _linearize(points, normals, fractions, target_points, target_normals, correspondences,
                          R_begin, t_begin, x[:3], x[3:], False)[2]

    _, b, _, used = _linearize(points, normals, fractions, target_points, target_normals, correspondences,
                               R_begin, t_begin, motion[:3], motion[3:], True)
    assert used == count

    numeric = np.zeros(6)
    for k in range(6):
        step = np.zeros(6)
        step[k] = 1e-6
        numeric[k] = (cost(motion + step) - cost(motion - step)) / 2e-6
    # The Gauss-Newton gradient ignores that the weights rotate with the points - a small term
    assert np.allclose(b, numeric, rtol=0.05, atol=1e-3 * np.abs(numeric).max())


@pytest.mark.parametrize('with_times', [False, True])
def test_motion_is_recovered_in_simulation(with_times):
    scan_list = scans(accelerating)
    if not with_times:
        scan_list = [(stamp, points, None) for stamp, points, _ in scan_list]
    trajectory = estimate(scan_list)

    # The map frame is the pose at the middle of the first scan (the identity: the lidar stands still then)
    for index, motion in enumerate(trajectory.motions):
        delta_trans, delta_rot = se3.pose_delta(motion.T_mid, pose_matrix(accelerating, (index + 0.5) * SCAN_PERIOD))
        assert delta_trans < 0.004, f'scan {index}: translation error {delta_trans}'
        assert np.degrees(delta_rot) < 0.03, f'scan {index}: rotation error {np.degrees(delta_rot)} deg'

    for t in (0.23, 0.61, 1.07):
        delta_trans, _ = se3.pose_delta(trajectory.pose_at(100.0 + t), pose_matrix(accelerating, t))
        assert delta_trans < 0.006

    # Every scan deskewed to some time of it lies on the walls when put there by the true pose of that time
    for index in (3, 8, 11):
        stamp, points, times = scan_list[index]
        fractions = point_time_fractions(times, points.shape[0], SCAN_PERIOD)
        for fraction in (0.0, 0.5, 1.0):
            reference = stamp + fraction * SCAN_PERIOD
            deskewed = trajectory.deskew(index, points, fractions, reference)
            T = pose_matrix(accelerating, reference - 100.0)
            world = deskewed @ T[:3, :3].T + T[:3, 3]
            assert np.percentile(distance_to_walls(world), 99) < 0.01

    # ... and without the deskewing it does not
    stamp, points, _ = scan_list[11]
    T = pose_matrix(accelerating, stamp - 100.0)
    assert np.percentile(distance_to_walls(points @ T[:3, :3].T + T[:3, 3]), 99) > 0.05


def test_recording_may_start_while_moving():
    """At 6 m/s the first scan is smeared by 60 cm; it must be unwarped with the motion of the second."""
    scan_list = scans(moving, num_scans=8)
    trajectory = estimate(scan_list)
    T_first_inverse = se3.inverse(pose_matrix(moving, 0.5 * SCAN_PERIOD))
    for index, motion in enumerate(trajectory.motions):
        delta_trans, delta_rot = se3.pose_delta(motion.T_mid, T_first_inverse @ pose_matrix(moving, (index + 0.5) * SCAN_PERIOD))
        assert delta_trans < 0.004, f'scan {index}: translation error {delta_trans}'
        assert np.degrees(delta_rot) < 0.03, f'scan {index}: rotation error {np.degrees(delta_rot)} deg'

    # The first scan is deskewed too
    stamp, points, times = scan_list[0]
    deskewed = trajectory.deskew(0, points, point_time_fractions(times, points.shape[0], SCAN_PERIOD), stamp)
    T = pose_matrix(moving, 0.0)
    assert np.percentile(distance_to_walls(deskewed @ T[:3, :3].T + T[:3, 3]), 99) < 0.01


def test_alignment_does_not_oscillate_when_standing():
    """A lidar that does not move must stay put, however the scene is distributed around it."""
    rng = np.random.default_rng(5)
    estimator = MotionEstimator(AlignmentOptions(min_distance=0.5))
    for index in range(10):
        # Only the first third of each scan sees the scene (the rest is removed, as by a near-range filter)
        azimuth = np.linspace(0.0, 2.0 * np.pi / 3.0, 8000)
        elevation = rng.uniform(-0.4, 0.4, azimuth.shape[0])
        directions = np.column_stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                                      np.sin(elevation)])
        distances = np.min(np.where(directions > 0, ROOM_HALF / np.maximum(directions, 1e-12),
                                    -ROOM_HALF / np.minimum(directions, -1e-12)), axis=1)
        points = directions * distances[:, None] + rng.normal(scale=0.01, size=directions.shape)
        estimator.add(points, point_time_fractions(None, points.shape[0], 0.1), 0.1 * index, 0.1)
    for motion in estimator.motions:
        delta_trans, delta_rot = se3.pose_delta(np.eye(4), motion.T_mid)
        assert delta_trans < 0.01
        assert np.degrees(delta_rot) < 0.05


def test_scans_without_points_continue_the_motion():
    scan_list = scans(accelerating, num_scans=10)
    empty = (scan_list[5][0], np.zeros((0, 3)), None)
    trajectory = estimate(scan_list[:5] + [empty] + scan_list[6:])
    assert not trajectory.motions[5].aligned
    # The scans after the gap are still found
    delta_trans, _ = se3.pose_delta(trajectory.motions[9].T_mid, pose_matrix(accelerating, 9.5 * SCAN_PERIOD))
    assert delta_trans < 0.01

    # Leading scans without points: the first scan with points becomes the reference
    trajectory = estimate([(scan_list[0][0], np.zeros((0, 3)), None)] + scan_list[1:6])
    assert len(trajectory.motions) == 6


def test_map_window_drops_what_is_not_seen_again():
    alignment_map = AlignmentMap(0.05, window_scans=3)
    old = np.array([[10.0, 0.0, 0.0]])
    kept = np.array([[0.0, 10.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0]])
    alignment_map.insert(np.vstack([old, kept]), np.vstack([normals, normals]), 0)
    for index in range(1, 5):
        alignment_map.insert(kept + 0.01, normals, index)  # sees `kept` again, adds nothing new
    assert alignment_map.points.shape[0] == 1
    assert np.allclose(alignment_map.points[0], kept[0])


def test_sample_evenly_covers_sparse_regions():
    rng = np.random.default_rng(3)
    dense = rng.uniform(0.0, 1.0, size=(50000, 3))
    sparse = rng.uniform(20.0, 40.0, size=(500, 3))
    indices = sample_evenly(np.concatenate([dense, sparse]), 1000, 0.5, rng)
    assert indices.shape[0] <= 1200
    # A plain random sample would take ~10 of the sparse points
    assert np.sum(indices >= 50000) > 400


def test_scan_durations_ignore_a_dropped_scan():
    stamps = np.array([0.0, 0.1, 0.2, 0.4, 0.5])
    assert np.allclose(scan_durations(stamps), [0.1, 0.1, 0.1, 0.1, 0.1])


def test_point_times_that_do_not_span_the_scan_are_ignored():
    fractions = point_time_fractions(np.zeros(100), 100, 0.1)
    assert np.allclose(fractions, np.arange(100) / 100)
    assert np.allclose(point_time_fractions(np.linspace(0, 0.1, 11), 11, 0.1), np.linspace(0, 1, 11))
