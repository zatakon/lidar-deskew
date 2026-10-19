"""Motion compensation (deskewing) of lidar scans, with the motion from the scans or from a GNSS track."""
from .deskew import DeskewOptions, deskew_recording
from .kml import GnssTrack, KmlOptions, KmlTrajectory, Mounting, load_kml
from .pointcloud2 import LidarFrame, decode_frame
from .scan_alignment import (AlignmentOptions, LidarTrajectory, MotionEstimator, ScanMotion, point_time_fractions,
                             scan_durations)

__version__ = '0.1.0'
