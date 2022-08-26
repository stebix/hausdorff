import importlib
import numpy as np

from functools import partial
from typing import Callable, Optional, Union, Tuple
from collections import namedtuple

from src.hausdorff_functional import (directed_hausdorff_distances,
                                      directed_hausdorff_distances_argtracked)
from src.parallel import parallel_compute_hausdorff, hardware_check
from src.utils import noop
from src.intersectiontools import argwhere, mask_unique_true, postpad


TrackedDistances = namedtuple(typename='TrackedDistances',
                              field_names=('distances', 'indices'))

POSTPROCESSING_FN = {
    'sqrt' : np.sqrt,
    'exp' : np.exp,
    'log' : np.log,
    'none' : noop
}

REDUCTION_FN = {
    'none' : noop,
    'max' : np.max,
    'canonical' : np.max,
    'average' : np.mean,
    'quantile' : np.quantile
}


def get_postprocessing_function(alias: str) -> callable:
    """Return callable postprocessing function from string alias"""
    try:
        fn = POSTPROCESSING_FN[alias]
    except KeyError as e:
        message = f'Invalid postprocessing function alias "{alias}"'
        raise ValueError(message) from e
    return fn

def get_reduction_function(alias: str) -> callable:
    try:
        fn = REDUCTION_FN[alias]
    except KeyError as e:
        message = f'Invalid reduction function alias: "{alias}"'
        raise ValueError(message) from e
    return fn

def get_metric(name: str) -> Callable:
    metric_module = importlib.import_module(name='metrics')
    try:
        metric = getattr(metric_module, name)
    except AttributeError:
        raise RuntimeError(f'Cannot retrieve metric with name "{name}"')
    return metric

def argtracked_max_reduction(distances: np.ndarray) -> int:
    return np.argmax(distances)

def argtracked_noop_reduction(distances: np.ndarray) -> slice:
    return np.s_[:]

def core_hausdorff_function(metric: Union[str, callable],
                            argtracked: bool) -> Callable:
    """
    Create the core Hausdorff computation function via insertion of the
    desired metric `d(a,b)` and the argument-tracking setting.

    Parameters
    ----------

    metric : str or callable
        The metric used for the Hausdorff distance computation. Can be a string
        indicating the metric function present in the `metrics.py` module
        or a numba `@numba.jit(nopython=True)` decorated callable with the signature
        `func(a,b)` for two vectors.

    argtracked : bool
        Flag to indicate wether the HD computation is performed with argument
        tracking.

    Returns
    -------

    hd_func : callable
        The core Hausdorff distance computation function with the call signature
        `hd_func(X, Y)` for two point sets X and Y.
    """
    if argtracked:
        hd_func = directed_hausdorff_distances_argtracked
    else:
        hd_func = directed_hausdorff_distances
    if isinstance(metric, str):
        metric_fn = get_metric(metric)
    elif callable(metric):
        metric_fn = metric
    else:
        raise TypeError(f'metric argument must be str or callable, got {type(metric)}')
    hd_func = partial(hd_func, metric=metric_fn)
    return hd_func


def split_result(result: np.ndarray, cast_indexarray: bool = True,
                 dtype=np.int32) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Split result of opaque Hausdorff computation function that may return a
    1D array (default, non-argtracked min-max computation) or a 2D array
    (specialized, argtracked min-max computation). The index array can be recast
    to the indicated dtype.
    """
    if result.ndim == 1:
        return (result, None)
    elif result.ndim == 2:
        distances, indices = result[:, 0], result[:, 1]
        if cast_indexarray:
            indices = indices.astype(dtype)
        return (distances, indices)
    else:
        message = (f'expecting 1D (non-argtracked) or 2D (argtracked) result argument, '
                   f'but got ndim = {result.ndim}')
        raise ValueError(message)


class _BaseHausdorff:
    """
    Interface to configure and then execute the Hausdorff distance computation.
    """
    repr_keys = set(('reduction', 'remove_intersection',
                     'metric', 'postprocess', 'parallelized',
                     'n_workers'))
    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: str = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 argtracked: bool = False,
                 parallelized: bool = False,
                 n_workers: Optional[int] = None,
                 ) -> None:
        self.reduction = reduction
        self.remove_intersection = remove_intersection
        self.metric = metric
        self.argtracked = argtracked
        self.parallelized = parallelized
        self.n_workers = n_workers

        if self.parallelized:
            assert n_workers, 'parallelization requires explicit setting of worker process count'
            hardware_check(n_workers)

        self.postprocess = postprocess


    def _get_core_hausdorff_function(self) -> callable:
        """
        Get the basal Hausdorff computation function in dependence of the
        desired settings.
        """
        return core_hausdorff_function(self.metric, self.argtracked)
    
    def _get_maybe_parallel_hausdorff_function(self) -> callable:
        """
        Get the potentially parallelized Hausdorff distance computation function.
        """
        core_hd_func = self._get_core_hausdorff_function()
        if self.parallelized:
            hd_func = partial(parallel_compute_hausdorff,
                              n_workers=self.n_workers, compute_func=core_hd_func,
                              argtracked=self.argtracked)
        else:
            hd_func = core_hd_func
        return hd_func
    
    def get_compute_fn(self) -> Callable:
        return self._get_maybe_parallel_hausdorff_function()
    
    def get_postprocessing_fn(self) -> Callable:
        return get_postprocessing_function(self.postprocess)
    
    def get_reduction_fn(self) -> Callable:
        if isinstance(self.reduction, str):
            return get_reduction_function(self.reduction)
        else:
            return self.reduction

    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        core_info = ', '.join(
            (f'{key}={self._wrap_strings(getattr(self, key))}' for key in self.repr_keys)
        )
        suffix = ')'
        return ''.join((prefix, core_info, suffix))
    
    @staticmethod
    def _wrap_strings(candidate):
        if isinstance(candidate, str):
            return ''.join(("'", candidate, "'"))
        else:
            return candidate


class DirectedHausdorff(_BaseHausdorff):

    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: str = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 parallelized: bool = False,
                 n_workers: Optional[int] = None) -> None:
        argtracked = False
        super().__init__(reduction, remove_intersection, metric, postprocess,
                         argtracked, parallelized, n_workers)

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Union[np.ndarray, float]:
        postprocess = self.get_postprocessing_fn()
        compute_HD = self.get_compute_fn()
        reduction = self.get_reduction_fn()
        args = (X, Y)

        intersection_count = 0
        if self.remove_intersection:
            (*args, intersection_count) = mask_unique_true(*args)
        args = tuple(argwhere(array) for array in args)
        distances = compute_HD(*args)
        distances = postpad(distances, pad_width=intersection_count)
        distances = postprocess(distances)
        distances = reduction(distances)
        return distances


class ArgtrackedDirectedHausdorff(_BaseHausdorff):
    reductions = ('none', 'max', 'canonical')
    def __init__(self,
                 reduction: str = 'none',
                 remove_intersection: bool = True,
                 metric: str = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 parallelized: bool = False,
                 n_workers: Optional[int] = None) -> None:
        assert reduction in self.reductions, f'unsupported reduction: must be one of {self.reductions}'
        argtracked = True
        super().__init__(reduction, remove_intersection, metric, postprocess,
                         argtracked, parallelized, n_workers)
    
    def get_reduction_fn(self) -> Callable:
        if self.reduction == 'none':
            return argtracked_noop_reduction
        else:
            return argtracked_max_reduction        

    def compute(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        postprocess = self.get_postprocessing_fn()
        compute_HD = self.get_compute_fn()
        reduction = self.get_reduction_fn()
        args = (X, Y)

        intersection_count = 0
        if self.remove_intersection:
            (*args, intersection_count) = mask_unique_true(*args)
        args = tuple(argwhere(array) for array in args)
        result = compute_HD(*args)
        result = postpad(result, pad_width=intersection_count)
        distances, indices = split_result(result, cast_indexarray=True, dtype=np.int32)
        distances = postprocess(distances)
        # select maximum index or full slice
        maxindex_or_slice: Union[int, slice] = reduction(distances)

        coordinates = (
            args[0][maxindex_or_slice, :],
            args[1][indices[maxindex_or_slice], :]
        )

        result = TrackedDistances(distances[maxindex_or_slice], coordinates)
        return result


class Hausdorff(_BaseHausdorff):

    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: str = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 parallelized: bool = False,
                 n_workers: Optional[int] = None) -> None:
        argtracked = False
        super().__init__(reduction, remove_intersection, metric, postprocess,
                         argtracked, parallelized, n_workers)

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Union[np.ndarray, float]:
        arg_combs = [(X, Y), (Y, X)]
        postprocess = self.get_postprocessing_fn()
        compute_HD = self.get_compute_fn()
        reduction = self.get_reduction_fn()
        # generalized result containers
        result_permutations = []
        intersection_count = 0
        for args in arg_combs:
            if self.remove_intersection:
                (*args, intersection_count) = mask_unique_true(*args)
            args = tuple(argwhere(array) for array in args)
            distances = compute_HD(*args)
            distances = postpad(distances, pad_width=intersection_count)
            distances = postprocess(distances)
            distances = reduction(distances)
            result_permutations.append(distances)
        if self.reduction != 'none':
            return np.max(result_permutations)
        else:
            return tuple(result_permutations)


class ArgtrackedHausdorff(_BaseHausdorff):

    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: str = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 parallelized: bool = False,
                 n_workers: Optional[int] = None) -> None:
        assert reduction in self.reductions, f'unsupported reduction: must be one of {self.reductions}'
        argtracked = True
        super().__init__(reduction, remove_intersection, metric, postprocess,
                         argtracked, parallelized, n_workers)

    def get_reduction_fn(self) -> Callable:
        if self.reduction == 'none':
            return argtracked_noop_reduction
        else:
            return argtracked_max_reduction    

    def compute(self, X: np.ndarray, Y: np.ndarray) -> Union[np.ndarray, float]:
        arg_combs = [(X, Y), (Y, X)]
        postprocess = self.get_postprocessing_fn()
        compute_HD = self.get_compute_fn()
        reduction = self.get_reduction_fn()
        # generalized result containers
        result_permutations = []
        indices_permutations = []
        intersection_count = 0
        for args in arg_combs:
            if self.remove_intersection:
                (*args, intersection_count) = mask_unique_true(*args)
            args = tuple(argwhere(array) for array in args)
            result = compute_HD(*args)
            result = postpad(*result, pad_width=intersection_count)
            distances, indices = split_result(result, cast_indexarray=True, dtype=np.int32)
            distances = postprocess(distances)
            maxindex_or_slice: Union[int, slice] = reduction(distances)
            result_permutations.append(distances[maxindex_or_slice])
            indices_permutations.append(indices[maxindex_or_slice])
        if self.reduction != 'none':
            # find maximum values per permutation
            maxindex = np.argmax(np.array(result_permutations))
            return TrackedDistances(result_permutations[maxindex], indices_permutations[maxindex])
        else:
            unreduced_result = tuple(
                TrackedDistances(dist_perm, idx_perm)
                for dist_perm, idx_perm in zip(result_permutations, indices_permutations)
            )
            return unreduced_result
