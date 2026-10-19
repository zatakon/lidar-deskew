"""Simulated scans of a lidar moving through a room, and recordings made of them."""
from datetime import datetime, timezone

import numpy as np

from lidar_deskew import se3

SCAN_PERIOD = 0.1
ROOM_HALF = np.array([12.0, 8.0, 3.0])


def accelerating(t):
    """Planar motion (yaw, x, y): stands for the first scan, then accelerates to 1.2 m/s turning with the distance."""
    t = np.asarray(t, dtype=np.float64)
    acceleration, top_speed = 1.2, 1.2
    since_start = np.maximum(t - SCAN_PERIOD, 0.0)
    ramp = top_speed / acceleration
    distance = np.where(since_start < ramp, 0.5 * acceleration * since_start ** 2,
                        0.5 * acceleration * ramp ** 2 + top_speed * (since_start - ramp))
    return 0.05 * distance, distance, 0.02 * distance ** 2


def accelerating_curve(t):
    """As `accelerating`, but driving where the car points to (as a car does, which a GNSS track assumes)."""
    yaw, distance, _ = accelerating(t)
    curvature = 0.05
    return yaw, np.sin(curvature * distance) / curvature, (1.0 - np.cos(curvature * distance)) / curvature


def moving(t, speed: float = 6.0):
    """Moves at `speed` [m/s] from the start, accelerating a little and turning."""
    t = np.asarray(t, dtype=np.float64)
    distance = speed * t + 0.25 * t * t
    return 0.03 * distance, distance - 6.0, 0.01 * distance ** 2


def pose_matrix(motion, t: float) -> np.ndarray:
    yaw, x, y = (float(np.asarray(value)) for value in motion(t))
    T = np.eye(4)
    T[:3, :3] = se3.exp_so3([0.0, 0.0, yaw])
    T[:3, 3] = [x, y, 0.0]
    return T


def simulated_scan(motion, start: float, points_per_scan: int, rng, T_vehicle_lidar=None):
    """
    Points (lidar frame) of one revolution starting at `start`, each measured from the pose at its own time,
    and their times since the start. `motion` is the motion of the vehicle, the lidar sits on it at
    `T_vehicle_lidar` (default: at its origin).
    """
    T_vl = np.eye(4) if T_vehicle_lidar is None else T_vehicle_lidar
    times = np.arange(points_per_scan) / points_per_scan * SCAN_PERIOD
    yaw, x, y = motion(start + times)
    azimuth = np.linspace(0.0, 2.0 * np.pi, points_per_scan, endpoint=False)
    elevation = rng.uniform(-0.4, 0.4, points_per_scan)
    directions = np.column_stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)])
    # Ray from the lidar in the world: vehicle pose * mounting
    cos, sin = np.cos(yaw), np.sin(yaw)

    def vehicle_to_world(vectors):
        return np.column_stack([cos * vectors[:, 0] - sin * vectors[:, 1], sin * vectors[:, 0] + cos * vectors[:, 1],
                                vectors[:, 2]])

    origins = vehicle_to_world(np.tile(T_vl[:3, 3], (points_per_scan, 1))) + np.column_stack([x, y, np.zeros_like(x)])
    world_directions = vehicle_to_world(directions @ T_vl[:3, :3].T)
    safe = np.where(np.abs(world_directions) < 1e-12, 1e-12, world_directions)
    distances = np.where(safe > 0, (ROOM_HALF - origins) / safe, (-ROOM_HALF - origins) / safe).min(axis=1)
    return directions * distances[:, None], times


def scans(motion, num_scans: int = 12, points_per_scan: int = 20000, seed: int = 2, stamp0: float = 100.0,
          T_vehicle_lidar=None):
    """List of (stamp, points, times) of consecutive scans; the stamp is the start of the scan."""
    rng = np.random.default_rng(seed)
    result = []
    for index in range(num_scans):
        points, times = simulated_scan(motion, index * SCAN_PERIOD, points_per_scan, rng, T_vehicle_lidar)
        result.append((stamp0 + index * SCAN_PERIOD, points, times))
    return result


def distance_to_walls(points_world: np.ndarray) -> np.ndarray:
    return np.min(np.abs(np.abs(points_world) - ROOM_HALF), axis=1)


# ---------------------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------------------
_FLOAT32, _UINT32 = 7, 6
POINT_STEP = 24   # x, y, z, intensity, t and 4 bytes nobody declares


def pointcloud2(types, stamp: float, points: np.ndarray, times=None, frame_id='lidar', ros1=False):
    PointField = types['sensor_msgs/msg/PointField']
    fields = [PointField(name=name, offset=4 * index, datatype=_FLOAT32, count=1)
              for index, name in enumerate(('x', 'y', 'z', 'intensity'))]
    count = points.shape[0]
    data = np.zeros((count, POINT_STEP), dtype=np.uint8)
    values = np.zeros((count, 4), dtype='<f4')
    values[:, :3] = points
    values[:, 3] = np.arange(count) % 255
    data[:, :16] = values.view(np.uint8).reshape(count, 16)
    if times is not None:
        fields.append(PointField(name='t', offset=16, datatype=_UINT32, count=1))
        data[:, 16:20] = np.round(times * 1e9).astype('<u4').view(np.uint8).reshape(count, 4)
    data[:, 20:24] = 0xAB  # padding that must survive
    seconds = int(np.floor(stamp))
    nanoseconds = int(round((stamp - seconds) * 1e9))
    Time = types['builtin_interfaces/msg/Time']
    header_arguments = dict(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id=frame_id)
    if ros1:
        header_arguments['seq'] = 0
    return types['sensor_msgs/msg/PointCloud2'](
        header=types['std_msgs/msg/Header'](**header_arguments), height=1, width=count, fields=fields,
        is_bigendian=False, point_step=POINT_STEP, row_step=POINT_STEP * count, data=data.reshape(-1),
        is_dense=True)


def write_mcap(path, scan_list, with_times=True, other_topic=True, topic='/points'):
    from mcap.writer import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_FOXY)
    with open(path, 'wb') as file:
        writer = Writer(file)
        writer.start(profile='ros2', library='test')
        schema = writer.register_schema('sensor_msgs/msg/PointCloud2', 'ros2msg',
                                        typestore.generate_msgdef('sensor_msgs/msg/PointCloud2', ros_version=2)[0].encode())
        channel = writer.register_channel(topic, 'cdr', schema)
        other = None
        if other_topic:
            other_schema = writer.register_schema('std_msgs/msg/String', 'ros2msg', b'string data')
            other = writer.register_channel('/note', 'cdr', other_schema)
        for index, (stamp, points, times) in enumerate(scan_list):
            msg = pointcloud2(typestore.types, stamp, points, times if with_times else None)
            log_time = int(round((stamp + SCAN_PERIOD) * 1e9))  # published at the end of the scan
            writer.add_message(channel, log_time, bytes(typestore.serialize_cdr(msg, 'sensor_msgs/msg/PointCloud2')),
                               log_time, index)
            if other is not None:
                note = typestore.types['std_msgs/msg/String'](data=f'note {index}')
                data = bytes(typestore.serialize_cdr(note, 'std_msgs/msg/String'))
                writer.add_message(other, log_time + 1000, data, log_time + 1000, index)
        writer.finish()


def write_bag(path, scan_list, topic='/points'):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Writer(path) as writer:
        connection = writer.add_connection(topic, 'sensor_msgs/msg/PointCloud2', typestore=typestore)
        for stamp, points, times in scan_list:
            msg = pointcloud2(typestore.types, stamp, points, times, ros1=True)
            writer.write(connection, int(round((stamp + SCAN_PERIOD) * 1e9)),
                         bytes(typestore.serialize_ros1(msg, 'sensor_msgs/msg/PointCloud2')))


def write_kml(path, motion, begin: float, end: float, clock_offset: float, period: float = 0.05,
              antenna_in_vehicle=(0.0, 0.0, 0.0), latitude=48.137, longitude=11.575):
    """KML track of the antenna; `motion` gives (yaw, east, north) over the lidar time, the GNSS clock is ahead."""
    times = np.arange(begin, end, period)
    yaw, east, north = motion(times)
    lever = np.asarray(antenna_in_vehicle)
    east = east + np.cos(yaw) * lever[0] - np.sin(yaw) * lever[1]
    north = north + np.sin(yaw) * lever[0] + np.cos(yaw) * lever[1]
    metres_per_degree_lat = 111132.0
    metres_per_degree_lon = 111320.0 * np.cos(np.radians(latitude))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>']
    for t, e, n in zip(times, east, north):
        when = datetime.fromtimestamp(t + clock_offset, tz=timezone.utc).isoformat(timespec='milliseconds')
        lines.append(f'<Placemark><TimeStamp><when>{when.replace("+00:00", "Z")}</when></TimeStamp>'
                     f'<Point><coordinates>{longitude + e / metres_per_degree_lon:.10f},'
                     f'{latitude + n / metres_per_degree_lat:.10f},500.0</coordinates></Point></Placemark>')
    lines.append('</Document></kml>')
    with open(path, 'w') as file:
        file.write('\n'.join(lines))
