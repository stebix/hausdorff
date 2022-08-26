"""
Various utility functions for the `hausdorff` package.
"""
import numpy as np


def is_binary(arr: np.array) -> bool:
    """
    Check if all array elements are approximately binary,
    i.e. elem € {0, 1} within float tolerance.

    Parameters
    ----------

    arr : numpy.ndarray
        The array to test.

    Returns
    -------

    is_binary : bool
        Boolean flag indicating the binary property.
    """
    bool_arr = np.logical_or(
        np.isclose(arr, 0), np.isclose(arr, 1)
    )
    if np.all(bool_arr):
        return True
    else:
        return False


def idshape(*arrays: np.ndarray) -> bool:
    const_shape = arrays[0].shape
    for arr in arrays:
        if arr.shape != const_shape:
            return False
    else:
        return True


def noop(x: np.ndarray):
    """No-op dummy function."""
    return x

