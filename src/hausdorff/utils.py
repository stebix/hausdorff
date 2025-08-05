"""
Various utility functions for the `hausdorff` package.
"""
import numpy as np

def all_isinstance(sequence, class_or_tuple) -> bool:
    """
    Check whether all elements of a sequence are instances of
    the given class or tuple of classes.
    """
    return all(isinstance(elem, class_or_tuple) for elem in sequence)

def unique_types(sequence) -> set:
    """Get the set of types of the objects present in the sequence."""
    return set(type(element) for element in sequence)

def wrap_strings(candidate, wrap_char="'"):
    """
    If candidate is a string, wrap it pre and post with the wrap char.
    Other types are returned unmodified.
    """
    if isinstance(candidate, str):
        return ''.join((wrap_char, candidate, wrap_char))
    return candidate


def dict_as_str_repr(dictionary) -> str:
    """
    Represent dictionary as a single comma-separated string. String values
    are automatically enclosed in single quotes. For a dictionary
    `{key_1 : value_1, key_2 : value_2, key_3 : string_value_3}` this
    yields the following result:
    "key_1=value_1, key_2=value_2, key_3='string_value_3'"
    """
    key, value = dictionary.popitem()
    core = f'{key}={wrap_strings(value)}'
    for key, value in dictionary.items():
        core = ', '.join((core, f'{key}={wrap_strings(value)}'))
    return core


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

