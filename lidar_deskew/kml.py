"""Motion of the lidar from a GNSS track (KML), and the comparison of a track with the motion from the scans.

A KML track holds positions only (latitude, longitude, altitude, typically 10-20 Hz), no
orientation. The orientation of the car is taken from the direction it drives in:

  - yaw: direction of the horizontal velocity of the (smoothed) track; the car drives forwards
  - pitch, roll: zero (the car is level)
  - below `min_speed` the direction is not defined; the last one is kept

The lidar follows the car by its mounting (`Mounting`): the pose of the lidar in the vehicle
frame (x forwards, y left, z up) and the position of the GNSS antenna in it.

This needs no structure in the scene, but it is only as good as these assumptions: driving
backwards, sliding or a car pitching over a bump are not modelled, and the GNSS clock may be
seconds off the lidar clock (see `compare_with_motion`, the `check-kml` command).
"""
import logging
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import se3

log = logging.getLogger(__name__)

# WGS84
_SEMI_MAJOR_AXIS = 6378137.0
_ECCENTRICITY_SQ = 6.69437999014e-3


@dataclass
class GnssTrack:
    stamps: np.ndarray      # (N,) [s] unix time (UTC) of the GNSS clock
    positions: np.ndarray   # (N,3) [m] east, north, up relative to the first point


def load_kml(filepath) -> GnssTrack:
    """Track from the placemarks of a KML file, each with a `TimeStamp/when` and a `Point/coordinates`."""
    root = ElementTree.parse(str(filepath)).getroot()

    def local_name(element) -> str:
        return element.tag.rsplit('}', 1)[-1]

    stamps, coordinates = [], []
    for placemark in root.iter():
        if local_name(placemark) != 'Placemark':
            continue
        when = coordinate_text = None
        for element in placemark.iter():
            name = local_name(element)
            if name == 'when':
                when = element.text.strip()
            elif name == 'coordinates':
                coordinate_text = element.text.strip()
        if when is None or coordinate_text is None:
            continue
        stamps.append(_parse_time(when))
        longitude, latitude, altitude = (float(value) for value in coordinate_text.split(',')[:3])
        coordinates.append((latitude, longitude, altitude))

    if len(stamps) < 2:
        raise ValueError(f'"{filepath}" has no track (placemarks with a time and a position)')

    order = np.argsort(stamps, kind='stable')
    stamps = np.array(stamps)[order]
    coordinates = np.array(coordinates)[order]

    # Without a position fix (e.g. indoors) some recorders write zeros
    has_fix = (np.abs(coordinates[:, 0]) > 1e-9) | (np.abs(coordinates[:, 1]) > 1e-9)
    if has_fix.sum() < 2:
        raise ValueError(f'"{filepath}" has no position fix (all coordinates are zero) - recorded indoors?')
    stamps, coordinates = stamps[has_fix], coordinates[has_fix]
    # The same time twice would make the velocity infinite: keep the first one
    unique = np.r_[True, np.diff(stamps) > 0]
    return GnssTrack(stamps=stamps[unique], positions=_to_east_north_up(coordinates[unique]))


def _parse_time(text: str) -> float:
    text = text.replace('Z', '+00:00')
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _to_east_north_up(coordinates: np.ndarray) -> np.ndarray:
    """Local tangent plane coordinates around the first point (exact enough for a few kilometres)."""
    latitude = np.radians(coordinates[:, 0])
    longitude = np.radians(coordinates[:, 1])
    sin_lat = np.sin(latitude[0])
    prime_vertical = _SEMI_MAJOR_AXIS / np.sqrt(1.0 - _ECCENTRICITY_SQ * sin_lat ** 2)
    meridian = _SEMI_MAJOR_AXIS * (1.0 - _ECCENTRICITY_SQ) / (1.0 - _ECCENTRICITY_SQ * sin_lat ** 2) ** 1.5
    east = (longitude - longitude[0]) * (prime_vertical + coordinates[0, 2]) * np.cos(latitude[0])
    north = (latitude - latitude[0]) * (meridian + coordinates[0, 2])
    up = coordinates[:, 2] - coordinates[0, 2]
    return np.column_stack([east, north, up])


# ---------------------------------------------------------------------------------------
# Motion of the lidar from the track
# ---------------------------------------------------------------------------------------
@dataclass
class Mounting:
    """
    Where the lidar and the GNSS antenna are on the car (vehicle frame: x forwards, y left, z up).

    The origin of the vehicle frame has to be on the axis the car turns around - the middle of the
    rear axle - as the car is taken to move where it points to there.
    """
    T_vehicle_lidar: np.ndarray = field(default_factory=lambda: np.eye(4))
    antenna_in_vehicle: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def from_values(lidar_xyz_rpy_deg=(0, 0, 0, 0, 0, 0), antenna_xyz=(0, 0, 0)) -> 'Mounting':
        x, y, z, roll, pitch, yaw = (float(value) for value in lidar_xyz_rpy_deg)
        T = np.eye(4)
        T[:3, :3] = se3.rot_from_rpy(np.radians(roll), np.radians(pitch), np.radians(yaw))
        T[:3, 3] = [x, y, z]
        return Mounting(T_vehicle_lidar=T, antenna_in_vehicle=np.asarray(antenna_xyz, dtype=np.float64))


@dataclass
class KmlOptions:
    time_offset: float = 0.0   # [s] GNSS clock - lidar clock
    smooth_s: float = 0.25     # [s] sigma of the Gaussian smoothing of the positions
    min_speed: float = 0.5     # [m/s] below it the driving direction is not taken from the track
    use_altitude: bool = True  # False: the car is taken to drive in a horizontal plane


def _heading(velocity: np.ndarray, min_speed: float) -> Optional[np.ndarray]:
    """Direction [rad, unwrapped] of the horizontal velocity, held while slower than `min_speed`; None if always."""
    moving = np.hypot(velocity[:, 0], velocity[:, 1]) >= min_speed
    if not moving.any():
        return None
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    # Hold the last known direction while standing, the first known one before it
    last_moving = np.maximum.accumulate(np.where(moving, np.arange(yaw.shape[0]), -1))
    last_moving[last_moving < 0] = np.flatnonzero(moving)[0]
    return np.unwrap(yaw[last_moving])


class KmlTrajectory:
    """Pose of the lidar over time from a GNSS track, in the east-north-up frame of the track."""

    def __init__(self, track: GnssTrack, mounting: Mounting = Mounting(), options: KmlOptions = KmlOptions()):
        self.mounting = mounting
        self.options = options

        # The track on an even grid in the lidar clock, smoothed
        period = float(np.median(np.diff(track.stamps)))
        lidar_stamps = track.stamps - options.time_offset
        self.grid = np.arange(lidar_stamps[0], lidar_stamps[-1] + 0.5 * period, period)
        positions = np.column_stack([np.interp(self.grid, lidar_stamps, track.positions[:, axis]) for axis in range(3)])
        if not options.use_altitude:
            positions[:, 2] = 0.0
        sigma = options.smooth_s / period
        if sigma > 0.0:
            positions = gaussian_filter1d(positions, sigma, axis=0, mode='nearest')
        self.positions = positions
        self.gaps = (lidar_stamps, np.diff(lidar_stamps))

        velocity = np.gradient(positions, period, axis=0)
        yaw = _heading(velocity, options.min_speed)
        if yaw is None:
            log.warning(f'The car never drives faster than {options.min_speed} m/s on the track - '
                        f'its direction is unknown, the lidar is taken not to turn')
            yaw = np.zeros(positions.shape[0])
        else:
            # The antenna moves where the car points to only if it sits on the axis the car turns around;
            # elsewhere it also moves by the turn (yaw rate x lever arm). Take that out, a few times.
            lever = mounting.antenna_in_vehicle
            for _ in range(3) if np.hypot(lever[0], lever[1]) > 0.0 else ():
                yaw_rate = np.gradient(yaw, period)
                cos, sin = np.cos(yaw), np.sin(yaw)
                lever_x, lever_y = cos * lever[0] - sin * lever[1], sin * lever[0] + cos * lever[1]
                origin_velocity = velocity[:, :2] - np.column_stack([-yaw_rate * lever_y, yaw_rate * lever_x])
                yaw = _heading(origin_velocity, options.min_speed)
        self.yaw = yaw

    def covers(self, begin: float, end: float) -> None:
        """Raises if the track does not cover the time span [s, lidar clock]."""
        first, last = self.grid[0], self.grid[-1]
        if begin < first or end > last:
            raise ValueError(
                f'The GNSS track ({first:.3f} - {last:.3f} in the lidar clock, with the offset '
                f'{self.options.time_offset:+.3f} s) does not cover the lidar scans ({begin:.3f} - {end:.3f}). '
                f'Check the clock offset (`check-kml`) or limit the time window.')
        stamps, gaps = self.gaps
        inside = (stamps[1:] > begin) & (stamps[:-1] < end)
        if inside.any() and gaps[inside].max() > 0.5:
            log.warning(f'The GNSS track has a gap of {gaps[inside].max():.2f} s during the lidar scans - '
                        f'the motion is interpolated over it')

    def vehicle_at(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Yaw [rad] and position [m] of the vehicle origin at the given times (lidar clock)."""
        times = np.asarray(times, dtype=np.float64)
        yaw = np.interp(times, self.grid, self.yaw)
        antenna = np.column_stack([np.interp(times, self.grid, self.positions[:, axis]) for axis in range(3)])
        lever = self.mounting.antenna_in_vehicle
        cos, sin = np.cos(yaw), np.sin(yaw)
        origin = antenna - np.column_stack([cos * lever[0] - sin * lever[1], sin * lever[0] + cos * lever[1],
                                            np.full_like(yaw, lever[2])])
        return yaw, origin

    def pose_at(self, stamp: float) -> np.ndarray:
        """Pose of the lidar at the given time (lidar clock) in the east-north-up frame of the track."""
        yaw, origin = self.vehicle_at(np.array([stamp]))
        T_world_vehicle = np.eye(4)
        T_world_vehicle[:3, :3] = se3.exp_so3([0.0, 0.0, yaw[0]])
        T_world_vehicle[:3, 3] = origin[0]
        return T_world_vehicle @ self.mounting.T_vehicle_lidar

    def deskew(self, points: np.ndarray, times: np.ndarray, reference_time: float) -> np.ndarray:
        """
        The points moved by the pose at their own times into the lidar frame at `reference_time`.

        @param times: (N,) absolute time of each point [s], lidar clock
        """
        R_vl, t_vl = self.mounting.T_vehicle_lidar[:3, :3], self.mounting.T_vehicle_lidar[:3, 3]
        yaw, origin = self.vehicle_at(times)
        yaw_reference, origin_reference = self.vehicle_at(np.array([reference_time]))

        in_vehicle = points @ R_vl.T + t_vl
        # Into the world by the pose of each point, and back by the pose of the reference: only the
        # differences to the reference matter, in the vehicle frame of the reference
        cos_r, sin_r = np.cos(yaw_reference[0]), np.sin(yaw_reference[0])
        delta = origin - origin_reference[0]
        shift = np.column_stack([cos_r * delta[:, 0] + sin_r * delta[:, 1], -sin_r * delta[:, 0] + cos_r * delta[:, 1],
                                 delta[:, 2]])
        turn = yaw - yaw_reference[0]
        cos, sin = np.cos(turn), np.sin(turn)
        in_reference_vehicle = np.column_stack([cos * in_vehicle[:, 0] - sin * in_vehicle[:, 1],
                                                sin * in_vehicle[:, 0] + cos * in_vehicle[:, 1],
                                                in_vehicle[:, 2]]) + shift
        return (in_reference_vehicle - t_vl) @ R_vl


# ---------------------------------------------------------------------------------------
# Comparison with the motion from the scans
# ---------------------------------------------------------------------------------------
def _speed(stamps: np.ndarray, positions: np.ndarray, smooth_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Speed [m/s] at the times between the samples, averaged over `smooth_s`."""
    times = 0.5 * (stamps[1:] + stamps[:-1])
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(stamps)
    period = float(np.median(np.diff(stamps)))
    window = max(1, int(round(smooth_s / period)))
    if window > 1:
        kernel = np.ones(window) / window
        speed = np.convolve(speed, kernel, mode='same')
    return times, speed


@dataclass
class GnssComparison:
    overlap_s: float
    time_offset_s: float            # GNSS clock - lidar clock, found by matching the two speed profiles
    speed_rms: float                # [m/s] difference of the speeds (with the offset applied)
    lidar_distance: float           # [m] travelled within the overlap
    gnss_distance: float
    times: np.ndarray
    lidar_speed: np.ndarray
    gnss_speed: np.ndarray

    def format(self) -> str:
        ratio = self.lidar_distance / self.gnss_distance if self.gnss_distance > 0.05 else float('nan')
        return '\n'.join([
            f'clock offset (GNSS - lidar) matching the speed profiles best: {self.time_offset_s:+.3f} s, '
            f'overlap {self.overlap_s:.1f} s',
            f'distance travelled: lidar {self.lidar_distance:.3f} m, GNSS {self.gnss_distance:.3f} m (ratio {ratio:.4f})',
            f'speed difference (RMS): {self.speed_rms:.3f} m/s',
        ])


def compare_with_motion(track: GnssTrack, lidar_stamps: np.ndarray, lidar_positions: np.ndarray,
                        smooth_s: float = 0.5) -> GnssComparison:
    """
    Compares the speed along a GNSS track with the speed of the lidar.

    The two clocks are not assumed to agree (the clock of a GNSS recorder can be seconds off): the
    time offset that matches the two speed profiles best is searched for first. The car has
    to change its speed during the recording for the offset to be defined.

    @param lidar_stamps: (N,) times [s] of the lidar positions
    @param lidar_positions: (N,3) positions of the lidar in its map frame
    """
    lidar_times, lidar_speed = _speed(lidar_stamps, lidar_positions, smooth_s)
    gnss_times, gnss_speed = _speed(track.stamps, track.positions, smooth_s)
    min_overlap = 0.5 * min(lidar_times[-1] - lidar_times[0], gnss_times[-1] - gnss_times[0])
    if min_overlap < 1.0:
        raise ValueError('The GNSS track or the lidar recording is shorter than two seconds')

    def rms_at(offset: float) -> float:
        """Difference of the speeds with the GNSS track read at lidar time + offset (inf if they barely overlap)."""
        start = max(lidar_times[0], gnss_times[0] - offset)
        end = min(lidar_times[-1], gnss_times[-1] - offset)
        if end - start < min_overlap:
            return np.inf
        times = lidar_times[(lidar_times >= start) & (lidar_times <= end)]
        return float(np.sqrt(np.mean((np.interp(times + offset, gnss_times, gnss_speed)
                                      - np.interp(times, lidar_times, lidar_speed)) ** 2)))

    coarse = np.arange(gnss_times[0] - lidar_times[-1], gnss_times[-1] - lidar_times[0], 0.05)
    coarse_best = coarse[int(np.argmin([rms_at(offset) for offset in coarse]))]
    fine = np.arange(coarse_best - 0.05, coarse_best + 0.05, 0.005)
    offset = float(fine[int(np.argmin([rms_at(candidate) for candidate in fine]))])

    start = max(lidar_times[0], gnss_times[0] - offset)
    end = min(lidar_times[-1], gnss_times[-1] - offset)
    times = lidar_times[(lidar_times >= start) & (lidar_times <= end)]
    lidar_on_grid = np.interp(times, lidar_times, lidar_speed)
    gnss_on_grid = np.interp(times + offset, gnss_times, gnss_speed)
    step = np.gradient(times)
    return GnssComparison(
        overlap_s=float(end - start), time_offset_s=offset, speed_rms=rms_at(offset),
        lidar_distance=float(np.sum(lidar_on_grid * step)), gnss_distance=float(np.sum(gnss_on_grid * step)),
        times=times, lidar_speed=lidar_on_grid, gnss_speed=gnss_on_grid,
    )


def forward_direction_in_lidar(stamps: np.ndarray, poses, min_speed: float = 1.5,
                               max_yaw_rate_deg: float = 3.0) -> Optional[Tuple[float, float, int]]:
    """
    Yaw and pitch [deg] of the lidar in the vehicle frame, from the direction the lidar moves in while the car
    drives straight on (assumed to be forwards). None if it never does.

    @param stamps: (N,) times of the poses
    @param poses: (N,4,4) poses of the lidar (map frame)
    @return: yaw, pitch [deg] for `Mounting.from_values`, and the number of samples used
    """
    directions = []
    for index in range(1, len(stamps) - 1):
        dt = stamps[index + 1] - stamps[index - 1]
        velocity = (poses[index + 1][:3, 3] - poses[index - 1][:3, 3]) / dt
        turn = np.linalg.norm(se3.log_so3(poses[index - 1][:3, :3].T @ poses[index + 1][:3, :3])) / dt
        if np.linalg.norm(velocity) < min_speed or np.degrees(turn) > max_yaw_rate_deg:
            continue
        in_lidar = poses[index][:3, :3].T @ velocity
        directions.append(in_lidar / np.linalg.norm(in_lidar))
    if not directions:
        return None
    forward = np.median(np.array(directions), axis=0)
    forward /= np.linalg.norm(forward)
    # The forward axis of the car in the lidar frame is R_vehicle_lidar^T e_x;
    # with R = Rz(yaw) Ry(pitch) that is (cos p cos y, -sin y, sin p cos y)
    yaw = np.arctan2(-forward[1], np.hypot(forward[0], forward[2]))
    pitch = np.arctan2(forward[2], forward[0])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), len(directions)
