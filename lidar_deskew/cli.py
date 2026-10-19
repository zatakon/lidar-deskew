"""Command line: `lidar-deskew deskew ...` and `lidar-deskew check-kml ...`."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .deskew import REFERENCES, DeskewOptions, deskew_recording, estimate_lidar_motion, read_scan_timings, scan_window
from .kml import KmlOptions, Mounting, compare_with_motion, forward_direction_in_lidar, load_kml
from .recording import Recording
from .scan_alignment import AlignmentOptions, scan_durations

log = logging.getLogger('lidar_deskew')


def _floats(count: int):
    def parse(text: str):
        values = [float(value) for value in text.replace(' ', ',').split(',') if value]
        if len(values) != count:
            raise argparse.ArgumentTypeError(f'expected {count} comma separated numbers, got "{text}"')
        return values
    return parse


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--topic', help='PointCloud2 topic (default: the only one)')
    parser.add_argument('--start', type=float, help='start of the time window [s from the first scan]')
    parser.add_argument('--end', type=float, help='end of the time window [s from the first scan]')
    parser.add_argument('--min-distance', type=float, default=AlignmentOptions.min_distance,
                        help='scan alignment: points closer to the lidar are not used [m] (default %(default)s)')
    parser.add_argument('--map-window', type=int, default=AlignmentOptions.map_window_scans,
                        help='scan alignment: map points not seen by the last N scans are dropped, 0 keeps all '
                             '(default %(default)s)')
    parser.add_argument('-v', '--verbose', action='store_true', help='log every scan')


def _alignment_options(arguments) -> AlignmentOptions:
    return AlignmentOptions(min_distance=arguments.min_distance, map_window_scans=arguments.map_window)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='lidar-deskew', description='Motion compensation of lidar scans')
    commands = parser.add_subparsers(dest='command', required=True)

    deskew = commands.add_parser('deskew', help='write the deskewed scans of a recording into an MCAP file',
                                 description='Moves every point by the motion of the lidar between the time it was '
                                             'measured and the reference time of its scan.')
    deskew.add_argument('input', type=Path, help='recording (.mcap or ROS1 .bag)')
    deskew.add_argument('output', type=Path, help='output .mcap')
    deskew.add_argument('--motion', choices=('lidar', 'kml'), default='lidar',
                        help='where the motion comes from: the scans themselves (default) or a GNSS track')
    deskew.add_argument('--kml', type=Path, help='GNSS track for --motion kml')
    deskew.add_argument('--kml-offset', type=float, default=0.0,
                        help='GNSS clock - lidar clock [s] (see check-kml), default 0')
    deskew.add_argument('--kml-smooth', type=float, default=KmlOptions.smooth_s,
                        help='sigma of the smoothing of the track [s] (default %(default)s)')
    deskew.add_argument('--kml-no-altitude', action='store_true', help='ignore the altitude of the track')
    deskew.add_argument('--lidar-in-vehicle', type=_floats(6), default=[0.0] * 6, metavar='X,Y,Z,ROLL,PITCH,YAW',
                        help='pose of the lidar in the vehicle frame (x forwards, y left, z up) [m, deg], '
                             'for --motion kml (default: aligned with the vehicle)')
    deskew.add_argument('--antenna-in-vehicle', type=_floats(3), default=[0.0] * 3, metavar='X,Y,Z',
                        help='position of the GNSS antenna in the vehicle frame [m], for --motion kml')
    deskew.add_argument('--reference', choices=REFERENCES, default='stamp',
                        help='time each scan is deskewed to: its header stamp (default, the stamp is kept), or the '
                             'start, middle or end of the scan (the stamp is set to it)')
    deskew.add_argument('--output-topic', help='topic of the deskewed scans (default: the input topic; with another '
                                               'name the original scans are kept too)')
    deskew.add_argument('--lidar-only', action='store_true', help='write only the deskewed scans, no other topics')
    deskew.add_argument('--trajectory', type=Path, help='also write the pose of the lidar per scan into this CSV')
    _add_common(deskew)

    check = commands.add_parser('check-kml', help='compare a GNSS track with the motion estimated from the scans',
                                description='Estimates the motion from the scans, finds the clock offset of the '
                                            'GNSS track by matching the speeds, and the direction of travel in '
                                            'the lidar frame - the parameters --motion kml needs.')
    check.add_argument('input', type=Path, help='recording (.mcap or ROS1 .bag)')
    check.add_argument('kml', type=Path, help='GNSS track')
    _add_common(check)
    return parser


def run_deskew(arguments) -> int:
    options = DeskewOptions(
        topic=arguments.topic, motion=arguments.motion, kml=arguments.kml,
        kml_options=KmlOptions(time_offset=arguments.kml_offset, smooth_s=arguments.kml_smooth,
                               use_altitude=not arguments.kml_no_altitude),
        mounting=Mounting.from_values(arguments.lidar_in_vehicle, arguments.antenna_in_vehicle),
        alignment=_alignment_options(arguments), reference=arguments.reference,
        output_topic=arguments.output_topic, copy_other_topics=not arguments.lidar_only,
        start_s=arguments.start, end_s=arguments.end,
    )
    deskew_recording(arguments.input, arguments.output, options, arguments.trajectory)
    return 0


def run_check_kml(arguments) -> int:
    track = load_kml(arguments.kml)
    print(f'GNSS track: {track.stamps.shape[0]} positions, {track.stamps[-1] - track.stamps[0]:.1f} s, '
          f'every {np.median(np.diff(track.stamps)) * 1000:.0f} ms')

    recording = Recording(arguments.input)
    topic = recording.find_topic('PointCloud2', arguments.topic)
    options = DeskewOptions(start_s=arguments.start, end_s=arguments.end)
    begin, end = scan_window(recording, topic, options)
    timings = read_scan_timings(recording, topic, begin, end)
    if len(timings) < 3:
        raise ValueError(f'Only {len(timings)} scans on "{topic}" in the time window')
    starts = np.array([timing.start for timing in timings])
    durations = scan_durations(starts)
    trajectory = estimate_lidar_motion(recording, topic, timings, durations, _alignment_options(arguments), begin, end)

    stamps = starts + 0.5 * durations
    poses = np.array([motion.T_mid for motion in trajectory.motions])
    print(compare_with_motion(track, stamps, poses[:, :3, 3]).format())

    forward = forward_direction_in_lidar(stamps, poses)
    if forward is None:
        print('direction of travel in the lidar frame: the car never drives straight on fast enough')
    else:
        yaw, pitch, count = forward
        print(f'direction of travel in the lidar frame (from {count} scans driving straight on, assumed forwards): '
              f'lidar yaw {yaw:+.2f} deg, pitch {pitch:+.2f} deg in the vehicle frame '
              f'-> --lidar-in-vehicle X,Y,Z,0,{pitch:.2f},{yaw:.2f}')
    return 0


def main(argv=None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('numba').setLevel(logging.WARNING)
    try:
        if arguments.command == 'deskew':
            return run_deskew(arguments)
        return run_check_kml(arguments)
    except (ValueError, FileNotFoundError) as error:
        log.error(str(error))
        return 1


if __name__ == '__main__':
    sys.exit(main())
