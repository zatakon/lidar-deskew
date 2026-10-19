"""Voxel grid that keeps one point per voxel."""
import numpy as np

# Bits per axis of a packed voxel coordinate: +-1 048 576 voxels, i.e. +-52 km at 5 cm
_BITS = 21
_OFFSET = 1 << (_BITS - 1)


def voxel_keys(points: np.ndarray, resolution: float) -> np.ndarray:
    """One int64 key per point, equal for the points of the same voxel."""
    coords = np.floor(points.astype(np.float64) / resolution).astype(np.int64)
    if coords.size and (coords.min() < -_OFFSET or coords.max() >= _OFFSET):
        raise ValueError(
            f'Points are too far from the origin for a voxel grid with the resolution {resolution} m '
            f'(the limit is {_OFFSET * resolution:.0f} m)'
        )
    coords += _OFFSET
    return (coords[:, 0] << (2 * _BITS)) | (coords[:, 1] << _BITS) | coords[:, 2]


def last_point_per_voxel(points: np.ndarray, resolution: float) -> np.ndarray:
    """
    Indices of the points that remain when each voxel keeps only the point inserted last.

    The indices are sorted, so the points stay in their insertion order.
    """
    keys = voxel_keys(points, resolution)
    # np.unique returns the first occurrence; on the reversed keys that is the last one
    _, first_in_reversed = np.unique(keys[::-1], return_index=True)
    indices = keys.shape[0] - 1 - first_in_reversed
    indices.sort()
    return indices

