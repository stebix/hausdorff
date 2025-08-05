"""
Provide tools to perform intersection-removal related operation before
the Hausdorff metric computation to reduce computational load.
"""
from typing import Tuple, Union, Sequence, TypeAlias

import numpy as np

from numpy.typing import NDArray

try:
    import cupy as cp
    HAS_CUPY = True
    Array: TypeAlias = Union[NDArray, cp.ndarray]
except ImportError:
    cp = None
    HAS_CUPY = False
    Array: TypeAlias = NDArray

def get_array_module(*arrays):
    """Get the appropriate array module (numpy or cupy) for the given arrays."""
    if HAS_CUPY:
        return cp.get_array_module(*arrays)
    return np

def all_ndim(*arrays, ndim: int) -> bool:
    """Check that all arrays are of given ndim."""
    return all(array.ndim == ndim for array in arrays)

def argwhere(array: Array) -> Array:
    """Array-module agnostic argwhere"""
    xp = get_array_module(array)
    return xp.argwhere(array)

def squeeze_singleton(arrays: Sequence[Array]) -> Union[Array, Tuple[Array]]:
    arrays = tuple(arrays)
    if len(arrays) != 1:
        return arrays
    return arrays[0]


def postpad(*arrays: Array, pad_width: int, squeeze: bool = True) -> Array:
    """
    Pad 1D or 2D arrays with pad_width zeros on the first axis.
    """
    # quickly handle no-op
    if pad_width == 0:
        if squeeze:
            return squeeze_singleton(arrays)
        else:
            return arrays
    xp = get_array_module(*arrays)
    if all_ndim(*arrays, ndim=2):
        pad_width = ((0, pad_width), (0, 0))
    elif all_ndim(*arrays, ndim=1):
        pad_width = ((0, pad_width),)
    else:
        ndims = [array.ndim for array in arrays]
        message = (f'postpad cannot operate on arrays with mixed ndim or '
                   f'ndim != 1 or 2, got arrays with ndim set [{ndims}]')
        raise ValueError(message)
    padded_arrays = tuple(
        xp.pad(array, mode='constant', pad_width=pad_width)
        for array in arrays
    )
    if squeeze:
        return squeeze_singleton(padded_arrays)
    return padded_arrays


def mask_unique_true(array_a: Array, array_b: Array,
                     return_secondary: bool = True,
                     return_intersection_count: bool = True,
                     squeeze: bool = True) -> Tuple:
    """
    Produce an array for two identically shaped boolean arrays where
    only elements True in array_a are set to True as well.
    Logical: result = (a_i and not b_i)

    Returns
    -------

    masked_array_a : Array
        A mask-like copy of the first array argument
        with the intersecting elements with the second
        array argument set to False.
    
    masked_array_b : Array, optional
        The unchanged secondary array argument.

    intersection_count : int, optional
        The number of elements in the intersection.
    """
    xp = get_array_module(array_a, array_b)
    intersection = xp.logical_and(array_a, array_b)
    # Avoid modifying supplied array object.
    array_a = array_a.copy()
    array_a[intersection] = False
    retval = [array_a]
    if return_secondary:
        retval.append(array_b)
    if return_intersection_count:
        intersection_count = xp.sum(intersection)
        retval.append(intersection_count)
    if squeeze and len(retval) == 1:
        return retval[0]
    return tuple(retval)


def compute_unique_points(array_a: Array, array_b: Array) -> Array:
    """
    Compute the array_a unique points (i.e. rows) for two 2D arrays
    array_a (N x D) and array_b (M x D).
    Hint: The arrays are interpreted as unordered sets of N and M
    D-dimensional points.

    Parameters
    ----------

    array_a : Array
        The first point set array of (N x D) dimensionality.

    array_b : Array
        The second point set array of (M x D) dimensionality. 
    """
    xp = get_array_module(array_a, array_b)
    concat_set = xp.concatenate((array_a, array_b), axis=0)
    return xp.unique(concat_set, axis=0)


def exclude_intersection(array_a: Array, array_b: Array) -> Array:
    """
    Return the foreground points that are uniquely contained in the first argument
    boolean array. This can be described as the array_a with the
    intersection with array_b excluded.

    Parameters
    ----------

    array_a : Array
        The first boolean array.
    
    array_b : Array
        The secondary boolean array.

    Returns
    -------

    unique_points : Array
        Array with foreground/True points uniquely contained
        in the first array. 
    """
    unique_points = compute_unique_points(argwhere(array_a), argwhere(array_b))
    return unique_points
