from types import SimpleNamespace

import numpy as np
import pytest

from lidar_deskew.pointcloud2 import decode_frame, point_times, with_points


def message(dtype, values):
    """A PointCloud2-like object with the given point fields."""
    datatypes = {np.dtype('<f4'): 7, np.dtype('<f8'): 8, np.dtype('<u4'): 6, np.dtype('<u2'): 4}
    array = np.zeros(len(values), dtype=dtype)
    for name in dtype.names:
        array[name] = values[name] if name in values.dtype.names else 0
    fields = [SimpleNamespace(name=name, offset=dtype.fields[name][1], datatype=datatypes[dtype.fields[name][0]], count=1)
              for name in dtype.names]
    return SimpleNamespace(fields=fields, is_bigendian=False, point_step=dtype.itemsize, width=len(values), height=1,
                           row_step=dtype.itemsize * len(values), data=np.frombuffer(array.tobytes(), dtype=np.uint8))


def structured(**columns):
    names = list(columns)
    array = np.zeros(len(next(iter(columns.values()))), dtype=[(name, np.asarray(columns[name]).dtype) for name in names])
    for name in names:
        array[name] = columns[name]
    return array


@pytest.mark.parametrize('name, values, stamp, start, span', [
    ('t', np.array([0, 50_000_000, 99_000_000], dtype='<u4'), 10.0, 10.0, 0.099),        # Ouster: ns since start
    ('time', np.array([-0.099, -0.05, 0.0], dtype='<f4'), 10.0, 9.901, 0.099),           # Velodyne: s, stamp at end
    ('timestamp', np.array([1.7e9 + 0.01, 1.7e9 + 0.06], dtype='<f8'), 1.7e9, 1.7e9 + 0.01, 0.05),  # Hesai: epoch s
    ('timestamp', np.array([1.7e18 + 1e7, 1.7e18 + 6e7], dtype='<f8'), 1.7e9, 1.7e9 + 0.01, 0.05),  # Livox: epoch ns
])
def test_point_time_conventions(name, values, stamp, start, span):
    data = structured(x=np.zeros(len(values), '<f4'), **{name: values})
    first, times = point_times(data, stamp)
    assert first == pytest.approx(start, abs=1e-6)
    assert times[0] == pytest.approx(0.0, abs=1e-9)
    assert times.max() == pytest.approx(span, abs=1e-6)


def test_no_point_times():
    data = structured(x=np.zeros(3, '<f4'))
    assert point_times(data, 5.0) == (5.0, None)


def test_points_are_replaced_and_everything_else_kept():
    dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity'], 'formats': ['<f4'] * 4, 'offsets': [0, 4, 8, 16],
                      'itemsize': 24})
    values = structured(x=np.array([1.0, np.nan], '<f4'), y=np.array([2.0, np.nan], '<f4'),
                        z=np.array([3.0, np.nan], '<f4'), intensity=np.array([7.0, 8.0], '<f4'))
    msg = message(dtype, values)
    raw = bytearray(msg.data.tobytes())
    raw[12:16] = raw[20:24] = b'\xab\xcd\xef\x01'          # bytes no field declares
    msg.data = np.frombuffer(bytes(raw), dtype=np.uint8)

    frame = decode_frame(msg, 1.0)
    assert np.allclose(frame.points[0], [1, 2, 3]) and np.isnan(frame.points[1]).all()
    moved = frame.points + 10.0
    data = with_points(msg, moved)
    assert data.dtype == np.uint8 and data.shape[0] == len(raw)
    result = np.frombuffer(data.tobytes(), dtype=dtype)
    assert np.allclose(result['x'][0], 11.0) and np.isnan(result['x'][1])
    assert np.array_equal(result['intensity'], [7.0, 8.0])
    assert data.tobytes()[12:16] == b'\xab\xcd\xef\x01' and data.tobytes()[20:24] == b'\xab\xcd\xef\x01'
