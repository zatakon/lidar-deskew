"""Deskewing of the scans of a recording: every point is moved by the motion of the lidar
between the time it was measured and the reference time of its scan.

    read the scan times  ->  motion (from the scans or from a KML track)  ->  write the scans

The output is an MCAP file with the deskewed scans in place of the original ones (same topic,
fields and layout; only x, y, z change) and, unless switched off, all other messages of the input.
"""
import csv
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import se3
from .kml import KmlOptions, KmlTrajectory, Mounting, load_kml
from .pointcloud2 import decode_frame, scan_start, with_points
from .recording import McapOutput, Recording, stamp_seconds
from .scan_alignment import AlignmentOptions, LidarTrajectory, MotionEstimator, point_time_fractions, scan_durations

log = logging.getLogger(__name__)

REFERENCES = ('stamp', 'start', 'middle', 'end')


@dataclass
class DeskewOptions:
    topic: Optional[str] = None           # the PointCloud2 topic, default the only one
    motion: str = 'lidar'                 # 'lidar': from the scans, 'kml': from a GNSS track
    kml: Optional[Path] = None
    kml_options: KmlOptions = field(default_factory=KmlOptions)
    mounting: Mounting = field(default_factory=Mounting)
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    # The time each scan is deskewed to: its header stamp (the stamp stays as it is), or its start,
    # middle or end (the stamp is set to that time)
    reference: str = 'stamp'
    output_topic: Optional[str] = None    # default: the input topic
    copy_other_topics: bool = True
    start_s: Optional[float] = None       # time window, seconds from the first scan
    end_s: Optional[float] = None


@dataclass
class ScanTiming:
    log_time: int        # [ns]
    stamp: float         # [s] header stamp
    start: float         # [s] time of the first point


def scan_window(recording: Recording, topic: str, options: DeskewOptions):
    """Log time window [ns] of the scans, from the window in seconds from the first scan."""
    if options.start_s is None and options.end_s is None:
        return None, None
    first = recording.first_log_time(topic)
    if first is None:
        raise ValueError(f'"{recording.filepath}" has no message on the topic "{topic}"')
    begin = None if options.start_s is None else first + int(round(options.start_s * 1e9))
    end = None if options.end_s is None else first + int(round(options.end_s * 1e9))
    return begin, end


def read_scan_timings(recording: Recording, topic: str, begin=None, end=None) -> List[ScanTiming]:
    timings = []
    for message in recording.messages([topic], begin, end):
        msg = recording.deserialize(message)
        stamp = stamp_seconds(msg.header.stamp)
        timings.append(ScanTiming(log_time=message.log_time, stamp=stamp, start=scan_start(msg, stamp)))
    return timings


def _progress(label: str, index: int, count: int, started: float) -> None:
    if (index + 1) % 100 == 0 or index + 1 == count:
        elapsed = time.perf_counter() - started
        log.info(f'{label}: scan {index + 1}/{count}, {(index + 1) / max(elapsed, 1e-9):.1f} scans/s')


def estimate_lidar_motion(recording: Recording, topic: str, timings: List[ScanTiming], durations: np.ndarray,
                          options: AlignmentOptions, begin=None, end=None) -> LidarTrajectory:
    estimator = MotionEstimator(options)
    started = time.perf_counter()
    for index, message in enumerate(recording.messages([topic], begin, end)):
        frame = decode_frame(recording.deserialize(message), timings[index].stamp)
        fractions = point_time_fractions(frame.times, len(frame), durations[index])
        valid = np.isfinite(frame.points).all(axis=1)
        estimator.add(frame.points[valid], fractions[valid], timings[index].start, durations[index])
        _progress('scan alignment', index, len(timings), started)
    motions = estimator.motions
    not_aligned = sum(not motion.aligned for motion in motions)
    if not_aligned:
        log.warning(f'{not_aligned} of {len(motions)} scans had too few points to be aligned')
    residuals = np.array([motion.residual for motion in motions if motion.num_correspondences > 0])
    if residuals.size:
        log.info(f'scan alignment: residual median {np.median(residuals) * 1000:.1f} mm, '
                 f'max {residuals.max() * 1000:.1f} mm')
    return LidarTrajectory(motions)


def _reference_time(reference: str, timing: ScanTiming, duration: float) -> float:
    if reference == 'stamp':
        return timing.stamp
    return timing.start + {'start': 0.0, 'middle': 0.5, 'end': 1.0}[reference] * duration


def _with_stamp(msg, stamp: float):
    seconds = int(np.floor(stamp))
    nanoseconds = int(round((stamp - seconds) * 1e9))
    if nanoseconds >= 1_000_000_000:
        seconds, nanoseconds = seconds + 1, nanoseconds - 1_000_000_000
    header_stamp = dataclasses.replace(msg.header.stamp, sec=seconds, nanosec=nanoseconds)
    return dataclasses.replace(msg, header=dataclasses.replace(msg.header, stamp=header_stamp))


def write_trajectory(path, timings: List[ScanTiming], durations: np.ndarray, trajectory) -> None:
    """Pose of the lidar at the middle of each scan, one row per scan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    motions = getattr(trajectory, 'motions', None)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['scan', 'start', 'duration', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        if motions is not None:
            header += ['aligned', 'correspondences', 'residual_m']
        writer.writerow(header)
        for index, (timing, duration) in enumerate(zip(timings, durations)):
            row = [index, f'{timing.start:.6f}', f'{duration:.6f}']
            row += [f'{value:.6f}' for value in se3.pose_to_values(trajectory.pose_at(timing.start + 0.5 * duration))]
            if motions is not None:
                motion = motions[index]
                row += [int(motion.aligned), motion.num_correspondences, f'{motion.residual:.4f}']
            writer.writerow(row)


def deskew_recording(input_path, output_path, options: DeskewOptions = DeskewOptions(),
                     trajectory_path=None) -> int:
    """Writes the deskewed scans of `input_path` into the MCAP file `output_path`. Returns the number of scans."""
    if options.reference not in REFERENCES:
        raise ValueError(f'Unknown reference "{options.reference}", expected one of {REFERENCES}')
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError('The output must not overwrite the input')
    recording = Recording(input_path)
    topic = recording.find_topic('PointCloud2', options.topic)
    output_topic = options.output_topic or topic
    begin, end = scan_window(recording, topic, options)

    timings = read_scan_timings(recording, topic, begin, end)
    if not timings:
        raise ValueError(f'No scans on "{topic}" in the time window')
    starts = np.array([timing.start for timing in timings])
    if np.any(np.diff(starts) <= 0.0):
        raise ValueError('The scans are not in the order of their times - reordered or duplicated messages?')
    durations = scan_durations(starts)
    log.info(f'{len(timings)} scans on "{topic}", {starts[-1] + durations[-1] - starts[0]:.1f} s, '
             f'every {np.median(durations) * 1000:.1f} ms')

    if options.motion == 'lidar':
        trajectory = estimate_lidar_motion(recording, topic, timings, durations, options.alignment, begin, end)
    elif options.motion == 'kml':
        if options.kml is None:
            raise ValueError('The motion from a KML track needs the KML file')
        trajectory = KmlTrajectory(load_kml(options.kml), options.mounting, options.kml_options)
        trajectory.covers(starts[0], starts[-1] + durations[-1])
    else:
        raise ValueError(f'Unknown motion source "{options.motion}", expected "lidar" or "kml"')

    if trajectory_path:
        write_trajectory(trajectory_path, timings, durations, trajectory)

    # Everything in the time window of the scans (only the scans without the other topics)
    topics = None if options.copy_other_topics else [topic]
    window_begin = timings[0].log_time if begin is not None else None
    window_end = timings[-1].log_time if end is not None else None
    keep_original = options.copy_other_topics and output_topic != topic

    index = 0
    started = time.perf_counter()
    with McapOutput(output_path, recording.profile) as output:
        for message in recording.messages(topics, window_begin, window_end):
            if message.topic != topic:
                output.write(message, message.data)
                continue
            if keep_original:
                output.write(message, message.data)
            timing, duration = timings[index], durations[index]
            if message.log_time != timing.log_time:
                raise RuntimeError('The scans were read in a different order than before')
            msg = recording.deserialize(message)
            frame = decode_frame(msg, timing.stamp)
            fractions = point_time_fractions(frame.times, len(frame), duration)
            reference_time = _reference_time(options.reference, timing, duration)

            valid = np.isfinite(frame.points).all(axis=1)
            points = frame.points.copy()
            if options.motion == 'lidar':
                points[valid] = trajectory.deskew(index, frame.points[valid], fractions[valid], reference_time)
            else:
                points[valid] = trajectory.deskew(frame.points[valid], timing.start + fractions[valid] * duration,
                                                  reference_time)

            msg = dataclasses.replace(msg, data=with_points(msg, points))
            if options.reference != 'stamp':
                msg = _with_stamp(msg, reference_time)
            output.write(message, recording.serialize(msg, message), topic=output_topic)
            _progress('writing', index, len(timings), started)
            index += 1
    if index != len(timings):
        raise RuntimeError(f'Wrote {index} scans, expected {len(timings)}')
    log.info(f'Wrote {index} deskewed scans to "{output_path}" ({output_topic})')
    return index
