"""
Provide tools to perform XOR operation before the Hausdorff metric
computation to reduce computational load.
"""
import numpy as np
import cupy as cp

from typing import Tuple, Union

Array = Union[np.ndarray, cp.ndarray]


def argwhere(array: Array) -> Array:
    """Array-module agnostic argwhere"""
    xp = cp.get_array_module(array)
    return xp.argwhere(array)


def exclude_intersection(map_a: np.ndarray, map_b: np.ndarray) -> Tuple:
    """
    Exclude the intersection of the two boolean segmentation maps in the first
    argument by setting the intersecting elements to False.

    Returns
    -------

    map_a : Array
        A copy of the first array argument with the intersecting
        elements with the second array argument set to False.

    intersection_count : int
        The number of elements in the intersection.
    """
    xp = cp.get_array_module(map_a, map_b)
    intersection = xp.logical_and(map_a, map_b)
    intersection_count = xp.sum(intersection)
    # Avoid modifying supplied array object.
    map_a = map_a.copy()
    map_a[intersection] = False
    return (map_a, intersection_count)
    