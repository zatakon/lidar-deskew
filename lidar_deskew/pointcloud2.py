"""Points and per-point times of sensor_msgs/PointCloud2 messages, and writing moved points back."""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# sensor_msgs/PointField datatypes
_DATATYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

# Names of the per-point time field, as different lidar drivers call it
TIME_FIELDS = ('t', 'time', 'time_stamp', 'timestamp')


@dataclass
class LidarFrame:
    """One scan of a lidar, with all points of the message (also the invalid ones, in their order)."""
    stamp: float                    # [s] header stamp
    points: np.ndarray              # (N,3) float64, lidar frame, may contain non-finite points
    start: float                    # [s] time of the first point (the header stamp without per-point times)
    times: Optional[np.ndarray]     # (N,) [s] time of each point since `start`, or None if the recording has none

    def __len__(self) -> int:
        return self.points.shape[0]


def point_dtype(msg) -> np.dtype:
    """Structured dtype of one point, with the fields of a single value (the others stay as padding)."""
    names, formats, offsets = [], [], []
    for field in msg.fields:
        if field.datatype not in _DATATYPES or field.count != 1:
            continue
        names.append(field.name)
        formats.append(('>' if msg.is_bigendian else '<') + _DATATYPES[field.datatype])
        offsets.append(field.offset)
    for axis in ('x', 'y', 'z'):
        if axis not in names:
            raise ValueError(f'The point cloud has no "{axis}" field, it has: {names}')
    return np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': msg.point_step})


def point_array(msg) -> np.ndarray:
    """The points of the message as a structured array (a read-only view of its data)."""
    num_points = int(msg.width) * int(msg.height)
    if msg.height > 1 and msg.row_step != msg.width * msg.point_step:
        raise ValueError(f'Rows with padding are not supported (row_step {msg.row_step}, '
                         f'width {msg.width} x point_step {msg.point_step})')
    return np.frombuffer(memoryview(msg.data), dtype=point_dtype(msg), count=num_points)


def _time_field(data: np.ndarray) -> Optional[str]:
    for name in TIME_FIELDS:
        if name in data.dtype.names:
            return name
    return None


def point_times(data: np.ndarray, stamp: float):
    """
    Time of the first point [s] and the times of all points relative to it, or (stamp, None).

    Drivers store the time differently; what the values look like tells which way it is:
      - uint32: nanoseconds since the start of the scan (Ouster, Livox ROS driver)
      - float > 1e16: nanoseconds since the epoch (Livox)
      - float > 1e6: seconds since the epoch (Hesai), or nanoseconds since the stamp
      - otherwise: seconds relative to the stamp (Velodyne ROS driver: may be negative, the stamp
        is then the end of the scan)
    """
    name = _time_field(data)
    if name is None or data.shape[0] == 0:
        return stamp, None
    kind = data.dtype.fields[name][0]
    times = data[name].astype(np.float64)
    finite = np.isfinite(times)
    if not finite.any():
        return stamp, None
    if kind.kind == 'u' and kind.itemsize == 4:
        times = times * 1e-9
    first, last = float(times[finite].min()), float(times[finite].max())

    if max(abs(first), abs(last)) > 1e16:
        times, first = times * 1e-9, first * 1e-9
        start = first
    elif max(abs(first), abs(last)) > 1e6:
        if abs(first - stamp) < 10.0:
            start = first
        elif (last - first) * 1e-9 < 1.0:
            times, first = times * 1e-9, first * 1e-9
            start = stamp + first
        else:
            raise ValueError(f'The point times (field "{name}", {first:.6g} .. {last:.6g}) do not fit the stamp '
                             f'{stamp:.6f} in any known unit')
    else:
        start = stamp + first
    return start, np.where(finite, times - first, 0.0)


def decode_frame(msg, stamp: float) -> LidarFrame:
    data = point_array(msg)
    points = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
    start, times = point_times(data, stamp)
    return LidarFrame(stamp=float(stamp), points=points, start=start, times=times)


def scan_start(msg, stamp: float) -> float:
    """Time of the first point of the scan, without decoding the points."""
    return point_times(point_array(msg), stamp)[0]


def with_points(msg, points: np.ndarray) -> np.ndarray:
    """A copy of the data of the message with its x, y, z replaced by `points` (N,3), as uint8."""
    # A copy of the structured array would lose the bytes no field declares: write through a view of the raw copy
    raw = np.array(np.frombuffer(memoryview(msg.data), dtype=np.uint8), copy=True)
    data = np.frombuffer(raw, dtype=point_dtype(msg), count=int(msg.width) * int(msg.height))
    for axis, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, axis]
    return raw
