"""
Convenience abstraction layer to compute Hausdorff distance and metric for segmentation results,
i.e. ground truth and label arrays.
"""
import types
import numpy as np
import numba
from functools import wraps
from typing import Callable, Tuple, List, Optional

from utils import is_binary, idshape

FUNC_TYPES = (types.FunctionType, numba.core.registry.CPUDispatcher)


def transform_data(*arrays: np.ndarray, threshold: float = 0.5,
                   dtype: Optional[np.dtype] = np.bool_) -> Tuple[np.ndarray]:
    """
    Transform segmentation data arrays to be directly digestable by
    the Hausdorff distance functions.
    Enables automatic binarization via thresholding and data type conversion.
    """
    transformed = []
    for arr in arrays:
        # possible binarization to {0, 1}
        if not is_binary(arr):
            arr = np.where(arr >= threshold, 1, 0)
        # datatype recast with special treatment for boolean type
        if dtype and dtype != arr.dtype:
            if dtype == np.bool_:
                arr = np.isclose(arr, 1.0)
            else:
                arr = arr.astype(dtype)
        transformed.append(arr)
    # squeeze singleton list
    if len(transformed) == 1:
        return transformed[0]
    else:
        return tuple(transformed)


def preprocess(*segarrays: np.ndarray) -> Tuple[np.ndarray]:
    """
    Transform a binary semantic segmentation pair < prediction, label> towards
    two coordinate vector arrays.
    """
    assert idshape(*segarrays), 'Mismatching segmentation arrays shape'
    segarrays = transform_data(*segarrays)
    if isinstance(segarrays, tuple):
        return tuple(np.argwhere(arr) for arr in segarrays)
    else:
        return np.argwhere(segarrays)


def segmentation(hd_func: Callable) -> Callable:
    """
    Decorator that adapts Hausdorff metric or distance functions for the
    segmentation result use case, i.e. a pair of ndarrays (prediction, label).
    """
    @wraps(hd_func)
    def wrapper(*args, **kwargs):
        (coords0, coords1) = preprocess(*args[:2])
        result = hd_func(coords0, coords1, **kwargs)
        return result
    return wrapper


def decorate_module_functions(module: types.ModuleType,
                              decorator: Callable) -> Tuple:
    """
    Decorate all function members of a module.
    """
    decorated_functions = []
    for name in dir(module):
        obj = getattr(module, name)
        # hausdorff module uses numba to compile functions 
        # -> we have to support multiple dispatch numba functions as well
        if isinstance(obj, FUNC_TYPES):
            decorated_functions.append((name, decorator(obj)))
    return tuple(decorated_functions)


# programmatically make all functions from the hausdorff.py module available here

import sys
import hausdorff as hd

current_module = sys.modules[__name__]
for (name, func) in decorate_module_functions(hd, segmentation):
    setattr(current_module, name, func)


