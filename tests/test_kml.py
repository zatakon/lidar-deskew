import numpy as np
import pytest

from helpers import SCAN_PERIOD, distance_to_walls, write_kml
from lidar_deskew import se3
from lidar_deskew.kml import (GnssTrack, KmlOptions, KmlTrajectory, Mounting, compare_with_motion,
                              forward_direction_in_lidar, load_kml)


def driving(t):
    """Drives at 8 m/s along a curve (yaw follows the direction of travel), in the lidar clock."""
    t = np.asarray(t, dtype=np.float64)
    yaw = 0.3 * t
    speed = 8.0
    # Velocity (cos yaw, sin yaw) * speed, integrated
    east = speed / 0.3 * np.sin(0.3 * t) - 5.0
    north = speed / 0.3 * (1.0 - np.cos(0.3 * t)) - 2.0
    return yaw, east, north


def test_load_kml(tmp_path):
    write_kml(tmp_path / 'track.kml', driving, 0.0, 2.0, clock_offset=1.7e9)
    track = load_kml(tmp_path / 'track.kml')
    assert track.stamps.shape[0] == 40
    assert track.stamps[0] == pytest.approx(1.7e9, abs=1e-3)
    speed = np.linalg.norm(np.diff(track.positions, axis=0), axis=1) / np.diff(track.stamps)
    assert np.allclose(speed, 8.0, rtol=0.01)


def test_kml_without_fix(tmp_path):
    (tmp_path / 'indoor.kml').write_text(
        '<kml><Document>' + ''.join(f'<Placemark><TimeStamp><when>2026-01-01T00:00:0{k}Z</when></TimeStamp>'
                                    f'<Point><coordinates>0,0,0</coordinates></Point></Placemark>' for k in range(3))
        + '</Document></kml>')
    with pytest.raises(ValueError, match='no position fix'):
        load_kml(tmp_path / 'indoor.kml')


def test_deskew_from_the_track(tmp_path):
    """A lidar mounted turned and offset on a car on a curve: the deskewed scan matches the scene."""
    from helpers import simulated_scan

    T_vehicle_lidar = np.eye(4)
    T_vehicle_lidar[:3, :3] = se3.rot_from_rpy(0.0, 0.0, np.radians(30.0))
    T_vehicle_lidar[:3, 3] = [1.2, 0.3, 1.5]
    antenna = (-0.5, 0.0, 1.6)
    clock_offset = 12.34
    write_kml(tmp_path / 'track.kml', driving, -1.0, 3.0, clock_offset, antenna_in_vehicle=antenna)
    trajectory = KmlTrajectory(load_kml(tmp_path / 'track.kml'),
                               Mounting(T_vehicle_lidar=T_vehicle_lidar, antenna_in_vehicle=np.array(antenna)),
                               KmlOptions(time_offset=clock_offset, smooth_s=0.0))
    trajectory.covers(0.0, 1.0)

    rng = np.random.default_rng(4)
    start = 0.5
    points, times = simulated_scan(driving, start, 20000, rng, T_vehicle_lidar)
    for reference in (start, start + 0.5 * SCAN_PERIOD, start + SCAN_PERIOD):
        deskewed = trajectory.deskew(points, start + times, reference)
        # Into the world by the true pose of the lidar at the reference time
        yaw, east, north = (float(value) for value in driving(reference))
        T_world_vehicle = np.eye(4)
        T_world_vehicle[:3, :3] = se3.exp_so3([0.0, 0.0, yaw])
        T_world_vehicle[:3, 3] = [east, north, 0.0]
        T = T_world_vehicle @ T_vehicle_lidar
        world = deskewed @ T[:3, :3].T + T[:3, 3]
        assert np.percentile(distance_to_walls(world), 99) < 0.01, reference

    # The raw scan is 80 cm long at 8 m/s
    yaw, east, north = (float(value) for value in driving(start))
    T_world_vehicle = np.eye(4)
    T_world_vehicle[:3, :3] = se3.exp_so3([0.0, 0.0, yaw])
    T_world_vehicle[:3, 3] = [east, north, 0.0]
    T = T_world_vehicle @ T_vehicle_lidar
    assert np.percentile(distance_to_walls(points @ T[:3, :3].T + T[:3, 3]), 90) > 0.1


def test_track_must_cover_the_scans(tmp_path):
    write_kml(tmp_path / 'track.kml', driving, 0.0, 2.0, clock_offset=0.0)
    trajectory = KmlTrajectory(load_kml(tmp_path / 'track.kml'))
    with pytest.raises(ValueError, match='does not cover'):
        trajectory.covers(1.5, 2.5)


def test_standing_car_keeps_its_direction():
    stamps = np.arange(0.0, 4.0, 0.05)
    # Drives north-east for 1 s, stands for 2 s, drives on
    distance = np.clip(stamps, 0, 1) * 3.0 + np.clip(stamps - 3.0, 0, None) * 3.0
    positions = np.column_stack([distance / np.sqrt(2), distance / np.sqrt(2), np.zeros_like(distance)])
    trajectory = KmlTrajectory(GnssTrack(stamps=stamps, positions=positions), options=KmlOptions(smooth_s=0.0))
    yaw, _ = trajectory.vehicle_at(np.array([0.5, 2.0, 3.5]))
    assert np.allclose(np.degrees(yaw), 45.0, atol=1.0)


def test_clock_offset_is_found_from_the_speeds():
    stamps = np.arange(0.0, 20.0, 0.1)
    speed = 1.0 + np.sin(stamps * 0.7) ** 2 * 3.0
    distance = np.cumsum(speed) * 0.1
    lidar = np.column_stack([distance, np.zeros_like(distance), np.zeros_like(distance)])
    gnss_stamps = np.arange(-5.0, 30.0, 0.05)
    gnss = np.column_stack([np.interp(gnss_stamps, stamps, distance), np.zeros_like(gnss_stamps),
                            np.zeros_like(gnss_stamps)])
    track = GnssTrack(stamps=gnss_stamps + 7.3, positions=gnss)
    comparison = compare_with_motion(track, stamps, lidar)
    assert comparison.time_offset_s == pytest.approx(7.3, abs=0.05)
    assert comparison.lidar_distance == pytest.approx(comparison.gnss_distance, rel=0.01)


@pytest.mark.parametrize('yaw_deg, pitch_deg', [(0.0, 0.0), (90.0, 0.0), (-12.0, 3.0)])
def test_forward_direction_gives_the_mounting(yaw_deg, pitch_deg):
    R_vehicle_lidar = se3.rot_from_rpy(0.0, np.radians(pitch_deg), np.radians(yaw_deg))
    stamps = np.arange(0.0, 2.0, 0.1)
    poses = []
    for t in stamps:
        T_world_vehicle = np.eye(4)
        T_world_vehicle[:3, 3] = [5.0 * t, 0.0, 0.0]
        T = np.eye(4)
        T[:3, :3] = R_vehicle_lidar
        poses.append(T_world_vehicle @ T)
    yaw, pitch, count = forward_direction_in_lidar(stamps, np.array(poses))
    assert count == len(stamps) - 2
    assert yaw == pytest.approx(yaw_deg, abs=1e-6)
    assert pitch == pytest.approx(pitch_deg, abs=1e-6)
