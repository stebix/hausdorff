"""
Provide helper functions to create testing-related volume arrays (H x W x D).
"""
import numpy as np
import skimage.morphology as morph

from typing import Tuple

def create_simple_testvolume(shape: tuple, pos_a: tuple, pos_b: tuple) -> Tuple[np.ndarray]:
    """Two point elements set `True` at given positions inside the volume."""
    volume = np.full(shape, fill_value=False)
    volume_a = volume
    volume_b = volume_a.copy()
    volume_a[pos_a] = True
    volume_b[pos_b] = True
    return (volume_a, volume_b)

def create_overlapping_cubes(as_bool: bool = False) -> Tuple[np.ndarray]:
    """Overlapping cubes."""
    cube = morph.cube(20)
    pad_width_a = ((20, 20), (15, 25), (15, 25))
    pad_width_b = ((20, 20), (25, 15), (25, 15))
    volumes = (np.pad(cube, pad_width_a), np.pad(cube, pad_width_b))
    if as_bool:
        volumes = tuple(v > 0 for v in volumes)
    return volumes