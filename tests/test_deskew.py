"""The whole run: recording in, deskewed MCAP out."""
import csv

import numpy as np
import pytest

from helpers import SCAN_PERIOD, accelerating, accelerating_curve, distance_to_walls, pose_matrix, scans, write_bag, write_kml, write_mcap
from lidar_deskew.cli import main
from lidar_deskew.deskew import DeskewOptions, deskew_recording
from lidar_deskew.kml import KmlOptions
from lidar_deskew.pointcloud2 import decode_frame
from lidar_deskew.recording import Recording, stamp_seconds
from lidar_deskew.scan_alignment import AlignmentOptions

STAMP0 = 100.0


def read_output(path, topic='/points'):
    recording = Recording(path)
    frames, raw = [], []
    for message in recording.messages([topic]):
        msg = recording.deserialize(message)
        frames.append(decode_frame(msg, stamp_seconds(msg.header.stamp)))
        raw.append((message, msg))
    return recording, frames, raw


def wall_error(frame, motion=accelerating):
    T = pose_matrix(motion, frame.stamp - STAMP0)
    return np.percentile(distance_to_walls(frame.points @ T[:3, :3].T + T[:3, 3]), 99)


@pytest.fixture(scope='module')
def scan_list():
    return scans(accelerating, num_scans=12, stamp0=STAMP0)


@pytest.mark.parametrize('reference', ['stamp', 'middle', 'end'])
def test_mcap_round_trip(tmp_path, scan_list, reference):
    write_mcap(tmp_path / 'in.mcap', scan_list)
    options = DeskewOptions(reference=reference, alignment=AlignmentOptions(min_distance=0.5))
    assert deskew_recording(tmp_path / 'in.mcap', tmp_path / 'out.mcap', options, tmp_path / 'trajectory.csv') == 12

    recording, frames, raw = read_output(tmp_path / 'out.mcap')
    assert recording.profile == 'ros2'
    assert set(recording.topics()) == {'/points', '/note'}
    assert len(frames) == 12
    shift = {'stamp': 0.0, 'middle': 0.5, 'end': 1.0}[reference] * SCAN_PERIOD
    for index, frame in enumerate(frames):
        assert frame.stamp == pytest.approx(scan_list[index][0] + shift, abs=1e-6)
        assert wall_error(frame) < 0.01, index
    # The other fields and the layout are kept
    message, msg = raw[5]
    assert msg.point_step == 24 and [field.name for field in msg.fields] == ['x', 'y', 'z', 'intensity', 't']
    data = msg.data.reshape(-1, 24)
    assert (data[:, 20:24] == 0xAB).all()
    assert message.log_time == int(round((scan_list[5][0] + SCAN_PERIOD) * 1e9))

    with open(tmp_path / 'trajectory.csv') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12 and float(rows[11]['x']) == pytest.approx(pose_matrix(accelerating, 1.15)[0, 3], abs=0.01)


def test_raw_scans_are_not_on_the_walls(tmp_path, scan_list):
    """The check of the test above means something: without deskewing the points are off."""
    write_mcap(tmp_path / 'in.mcap', scan_list)
    _, frames, _ = read_output(tmp_path / 'in.mcap')
    assert wall_error(frames[11]) > 0.05


def test_bag_without_point_times_lidar_only_and_window(tmp_path, scan_list):
    write_bag(tmp_path / 'in.bag', [(stamp, points, None) for stamp, points, _ in scan_list])
    options = DeskewOptions(alignment=AlignmentOptions(min_distance=0.5), copy_other_topics=False,
                            output_topic='/points_deskewed', start_s=0.25, end_s=0.95)
    assert deskew_recording(tmp_path / 'in.bag', tmp_path / 'out.mcap', options) == 7

    recording, frames, raw = read_output(tmp_path / 'out.mcap', '/points_deskewed')
    assert set(recording.topics()) == {'/points_deskewed'}
    assert raw[0][0].encoding == 'ros1' and recording.profile == 'ros1'
    assert [round(frame.stamp - STAMP0, 3) for frame in frames] == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # The window starts while the car accelerates: the first scan only has the motion of the second one
    assert wall_error(frames[0]) < 0.02
    for frame in frames[1:]:
        assert wall_error(frame) < 0.01


def test_motion_from_kml(tmp_path):
    def vehicle(t):
        return accelerating_curve(np.asarray(t) - STAMP0)

    write_mcap(tmp_path / 'in.mcap', scans(accelerating_curve, num_scans=12, stamp0=STAMP0))
    write_kml(tmp_path / 'track.kml', vehicle, STAMP0 - 1.0, STAMP0 + 3.0, clock_offset=7.5)
    options = DeskewOptions(motion='kml', kml=tmp_path / 'track.kml',
                            kml_options=KmlOptions(time_offset=7.5, smooth_s=0.0, min_speed=0.05))
    deskew_recording(tmp_path / 'in.mcap', tmp_path / 'out.mcap', options)
    _, frames, _ = read_output(tmp_path / 'out.mcap')
    assert len(frames) == 12
    for frame in frames:
        assert wall_error(frame, accelerating_curve) < 0.01


def test_command_line(tmp_path, scan_list, capsys):
    write_mcap(tmp_path / 'in.mcap', scan_list, other_topic=False)
    assert main(['deskew', str(tmp_path / 'in.mcap'), str(tmp_path / 'out.mcap'), '--min-distance', '0.5',
                 '--reference', 'start']) == 0
    _, frames, _ = read_output(tmp_path / 'out.mcap')
    assert len(frames) == 12 and wall_error(frames[-1]) < 0.01

    assert main(['deskew', str(tmp_path / 'in.mcap'), str(tmp_path / 'in.mcap')]) == 1
    assert main(['deskew', str(tmp_path / 'in.mcap'), str(tmp_path / 'x.mcap'), '--motion', 'kml']) == 1
