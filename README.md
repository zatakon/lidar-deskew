# lidar-deskew

Motion compensation (deskewing) of lidar scans. A spinning lidar needs ~0.1 s per revolution; when the
car moves, every point is measured from a different pose and the scan is smeared (8 cm at 3 m/s, plus
the rotation). This tool moves every point by the motion of the lidar between the time it was
measured and one reference time of its scan.

- **Input:** a recording with `sensor_msgs/PointCloud2` scans - MCAP (ROS2 or ROS1 encoding) or a ROS1
  `.bag`. No ROS installation is needed.
- **Motion** from one of:
  - `lidar` (default): estimated from the scans themselves (scan-to-map GICP with a constant velocity
    within a scan). No other sensor, works indoors, needs some structure around the car.
  - `kml`: a GNSS track (KML with timestamped positions, e.g. 20 Hz). Position from the track, heading from
    the direction of travel; needs the mounting of the lidar and the clock offset of the track.
- **Output:** an MCAP file with the deskewed scans in place of the original ones - same topic, fields,
  layout and encoding, only `x, y, z` change - and all other messages of the input.

The scan alignment runs scan by scan with a map of the last scans only, so that long recordings can be
processed.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -e .                # dependencies: numpy, scipy, numba, rosbags, mcap
```

## Usage

```
# Motion from the scans
lidar-deskew deskew recording.mcap deskewed.mcap

# Only a time window, only the lidar topic, and the trajectory as CSV
lidar-deskew deskew recording.bag deskewed.mcap --start 10 --end 40 --lidar-only --trajectory poses.csv

# Motion from a GNSS track
lidar-deskew deskew recording.mcap deskewed.mcap --motion kml --kml track.kml --kml-offset 2.5 \
    --lidar-in-vehicle 1.2,0,1.9,0,0,0 --antenna-in-vehicle 0.5,0,1.6

# Clock offset of a track and the direction of travel in the lidar frame (motion estimated from the scans)
lidar-deskew check-kml recording.mcap track.kml --end 30
```

`python -m lidar_deskew ...` works as well. `lidar-deskew deskew --help` lists all options.

### Options worth knowing

| Option | Meaning |
| --- | --- |
| `--topic` | PointCloud2 topic; default: the only one in the recording |
| `--reference stamp\|start\|middle\|end` | time each scan is deskewed to. `stamp` (default): the header stamp, which stays as it is - what TF lookups and other tools assume. Otherwise the start, middle or end of the scan, and the header stamp is set to that time |
| `--output-topic` | write the deskewed scans under another name; the original scans are then kept too |
| `--lidar-only` | write only the deskewed scans |
| `--start`, `--end` | time window in seconds from the first scan (other topics are cut to it too) |
| `--min-distance` | scan alignment ignores closer points (the car itself), default 3 m |
| `--map-window` | scan alignment: map points no scan has seen for this many scans are dropped (default 30, 0: keep all) |
| `--trajectory` | CSV with the pose of the lidar at the middle of every scan (and the alignment residual) |

### Time of the points

The time of every point comes from the point field `t`, `time`, `time_stamp` or `timestamp` when there
is one. The unit and the reference are recognized from the values: `uint32` nanoseconds since the scan
start (Ouster, Livox ROS driver), seconds relative to the header stamp (Velodyne ROS driver, also
negative), seconds or nanoseconds since the epoch (Hesai, Livox). Without such a field the points are
taken to be in the order they were measured, evenly spread over the scan, and the header stamp to be
the start of the scan. A time field that spans less than 10 % of the scan (e.g. all zeros) is ignored
with a warning.

The duration of a scan is the time to the start of the next one.

### Motion from a KML track

The vehicle frame is x forwards, y left, z up, with its origin in the middle of the **rear axle**: there
the car moves where it points to, which is how the heading is taken from the track. Give

- `--lidar-in-vehicle X,Y,Z,ROLL,PITCH,YAW` - pose of the lidar in the vehicle frame [m, deg, URDF
  convention R = Rz(yaw) Ry(pitch) Rx(roll)]. The rotation matters most: it decides in which direction
  of the lidar frame the motion is taken out. `check-kml` measures yaw and pitch from the scans.
- `--antenna-in-vehicle X,Y,Z` - position of the GNSS antenna [m]; when the car turns, the antenna moves
  differently from the rear axle.
- `--kml-offset` - GNSS clock minus lidar clock [s]. The clock of a GNSS recorder can be seconds off
  the lidar clock. `check-kml` finds it by matching the speed from the track with the speed from the
  scan alignment; the car has to change its speed for that.

Assumed: the car drives forwards, does not slide and stays level (no pitch/roll from the track). The
track is smoothed (`--kml-smooth`, sigma 0.25 s); below 0.5 m/s the last direction is kept.
`--kml-no-altitude` ignores the altitude of the track. A track without a fix (zeros, recorded indoors) is
rejected. Accuracy is the accuracy of the track's velocity over 0.1 s - with a plain GNSS receiver
expect centimetres; prefer `lidar` wherever the scene has structure.

## Python API

```python
from lidar_deskew import DeskewOptions, deskew_recording, AlignmentOptions

deskew_recording('in.mcap', 'out.mcap', DeskewOptions(reference='middle', alignment=AlignmentOptions(min_distance=2.0)))
```

Or on scans already in memory:

```python
from lidar_deskew import MotionEstimator, point_time_fractions, LidarTrajectory

estimator = MotionEstimator()
for start, duration, points, times in my_scans:          # points (N,3) finite, times since start or None
    estimator.add(points, point_time_fractions(times, len(points), duration), start, duration)
trajectory = LidarTrajectory(estimator.motions)
deskewed = trajectory.deskew(index, points, fractions, reference_time)   # lidar frame at reference_time
pose = trajectory.pose_at(t)                                             # 4x4, frame of the first scan
```

`KmlTrajectory(load_kml(path), Mounting.from_values(...), KmlOptions(...)).deskew(points, abs_times, t_ref)`
does the same from a track.

## Limits

- Constant velocity between the middles of two scans. Under hard braking (3 m/s^2) the error is about 4 mm.
- The first scan of a recording (or time window) has only the motion of the second one; if the car
  accelerates there, it is less exact (1 cm in the tests).
- Scans with fewer than 100 points beyond `--min-distance` are not aligned; the motion is continued.
- The scan alignment needs structure: in an open field or in a tunnel with smooth walls the motion along
  the free direction is held only by the motion prior. Use `kml` there.
- Moving objects are not detected; they are deskewed with the motion of the lidar like everything else.
- Speed: about 2 scans/s for the scan alignment of 86 000-point scans with 4 threads (the KML motion is
  much faster).
- PointCloud2 with padded rows (`row_step > width * point_step`) is not supported.

## Tests

```
pip install -e .[dev]
python -m pytest
```

Simulated scans of a lidar moving through a room (each point measured from the pose at its own time):
the deskewed points must lie on the walls within 1 cm (99 %), for both motion sources, for MCAP and bag
inputs, with and without per-point times.
