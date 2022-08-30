import importlib
import dataclasses
import numpy as np

from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple, Dict, List
from collections import namedtuple

from hausdorff.functional import (directed_hausdorff_distances,
                                  directed_hausdorff_distances_argtracked)
from hausdorff.parallel import parallel_compute_hausdorff, hardware_check, parallel_compute_reduction
from hausdorff.utils import all_isinstance, noop, dict_as_str_repr, unique_types
from hausdorff.intersectiontools import argwhere, mask_unique_true, postpad


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
    'mean' : np.mean,
    'quantile' : np.quantile,
    'median' : np.median
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
    metric_module = importlib.import_module(name='hausdorff.metrics')
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
                 metric: Union[str, Callable] = 'squared_euclidean',
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

    def _get_core_attr_dict(self) -> dict:
        """
        Assemble a dictionary containing core attribute names and values.
        The attributes deemed as "core" are specified in the `repr_keys`
        class variable. 
        """
        return {key : getattr(self, key) for key in self.repr_keys}

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
        """
        Wrap any string-type argument with pre- and post single quotes,
        i.e. input is transformed like: foobar -> 'foobar'
        Arguments of any other type are returned unmodified.
        """
        if isinstance(candidate, str):
            return ''.join(("'", candidate, "'"))
        else:
            return candidate


class DirectedHausdorff(_BaseHausdorff):
    """
    Compute the directed Hausdorff distance between the first volume `X` and
    the second volume `Y`.
    The computation algorithm can be modified via the class initializer arguments.
    The computation is performed via the `compute(X, Y)` method.

    Parameters
    ----------

    reduction : string or callable, optional
        The reduction is applied to the array of minimal distances
        for every point in `X` to any point in `Y`.
        Use 'none' to get the full array.
        Use 'max' to get the canonical directed Hausdorff distance.
        Use 'average' to get the average directed Hausdorff distance.
        Alternatively, any supplied callable is applied to the array
        of minimal distances.
        Defaults to 'none'.

    remove_intersection : bool, optional
        Exclude intersecting points in `X` and `Y` to reduce
        the computational load.
        Defaults to True.
    """

    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: Union[str, Callable] = 'squared_euclidean',
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
                 metric: Union[str, Callable] = 'squared_euclidean',
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
    """Compute the canonical, symmetric Hausdorff distance."""
    def __init__(self,
                 reduction: Union[str, callable] = 'none',
                 remove_intersection: bool = True,
                 metric: Union[str, Callable] = 'squared_euclidean',
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
                 metric: Union[str, Callable] = 'squared_euclidean',
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



@dataclasses.dataclass
class ReductionSpec:
    """Specify a reduction function via its string name and a kwarg dict."""
    name: str
    kwargs: Dict = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return f"({self.name}, {self.kwargs})"
    
    def asdict(self) -> dict:
        """Return instance as dictionary."""
        return {'name' : self.name, 'kwargs' : self.kwargs}
    
    @classmethod
    def from_dict(cls, dictionary: dict) -> 'ReductionSpec':
        """Initialize instance from a flat dictionary."""
        name = dictionary.pop('name')
        return cls(name=name, kwargs=dictionary)


class MultireductionHausdorff:
    """
    Efficiently computes symmetric/canonical Hausdorff distance with a number
    of reductions. 

    Parameters
    ----------

    reduction_specs : Sequence of dictionary or ReductionSpec
        Specify the multiple reduction functions via a sequence
        of dictionaries or ReductionSpec objects.

    remove_intersection : bool, optional
        Exclude matching or intersecting points (that would yield zero
        distance in the distance matrix) to mitigate compute load.
        Defaults to True.

    metric: str or callable, optional
        The desired metric to compute point to point distance.
        Can be a string name/alias or a `@numba.jit(nopython=True)`
        decorated function.
        Defaults to 'squared_euclidean'
    
    postprocess : str, optional
        Element-wise postprocessing function applied to the distance
        matrix. Select the operation via a string alias/name.
        Typically used in conjunction with the metric selection
        to mitigate computational load.
        Defaults to 'sqrt'.
    
    parallelized : bool, optional
        Toggle parallelization at Hausdorff distance matrix
        computation.
        Defaults to False.
    
    n_workers : int, optional
        Set the maximum worker process count. Only utilized for
        `parallelized=True`.
        Defaults to None.

    parallelized_reduction : bool, optional
        Toggle parallelized reduction of both directed Hausdorff
        distance matrix permutations.
        Defaults to False.
    """
    def __init__(self,
                 reduction_specs: Sequence[Union[Dict, ReductionSpec]],
                 remove_intersection: bool = True,
                 metric: Union[str, Callable] = 'squared_euclidean',
                 postprocess: str = 'sqrt',
                 parallelized: bool = False,
                 n_workers: Optional[int] = None,
                 parallelized_reduction: bool = False) -> None:

        reduction = 'none'
        self.base_hausdorff = Hausdorff(reduction, remove_intersection,
                                        metric, postprocess,
                                        parallelized, n_workers)
        self.parallelized_reduction = parallelized_reduction
        self.reduction_specs = self.initialize_reduction_specs(reduction_specs)
        self.reductions = self.create_reduction_functions(self.reduction_specs)
    
    @staticmethod
    def initialize_reduction_specs(reduction_specs: Sequence[Union[dict, ReductionSpec]]) -> List[ReductionSpec]:
        """
        Check type consistency of argument sequence and conditionally create
        a list of ReductionSpec instances if a sequence of dictionaries is
        provided.
        """
        if all_isinstance(reduction_specs, ReductionSpec):
            return reduction_specs
        elif all_isinstance(reduction_specs, dict):
            return [ReductionSpec.from_dict(elem_dict) for elem_dict in reduction_specs]
        else:
            message = (f'reduction_spec sequence must be type-homogenous with either {type(dict)} '
                       f'or {type(ReductionSpec)}. Error due to sequence containing '
                       f'multiple types: {unique_types(reduction_specs)}')
            raise TypeError(message)

    @staticmethod
    def create_reduction_functions(reduction_specs: Sequence[ReductionSpec]) -> List[Callable]:
        """Create reduction callable functions from sequence of ReductionSpec instances."""
        reductions = []
        for reduction_spec in reduction_specs:
            fn_reference = get_reduction_function(reduction_spec.name)
            # partial to bind kwargs such that func(array) is guaranteed to work
            func = partial(fn_reference, **reduction_spec.kwargs)
            reductions.append(func)
        return reductions
    
    def compute_reduction(self,
                          permutation_A: np.ndarray, permutation_B: np.ndarray,
                          func: Callable) -> float:
        """
        Compute the reduction for the two directed Hausdorff distance permutations.
        The implementation is chosen upon the desired parallelization behaviour.
        """
        # TODO: Typical reduction ufuncs probably release the GIL anyway such that the
        # process pool overhead kills any gains at typical permutation array
        # sizes. Confirming and optimizing this requires more in-depth knowledge of numpy 
        # parallelism though.
        if self.parallelized_reduction:
            result = parallel_compute_reduction(permutation_A, permutation_B, func)
        else:
            result_A = func(permutation_A)
            result_B = func(permutation_B)
            result = (result_A, result_B)
        return np.max(result)

    def compute(self, X: np.ndarray, Y: np.ndarray) -> List[Dict]:
        results = []
        dirHD_XY, dirHD_YX = self.base_hausdorff.compute(X, Y)
        for reduction_func, reduction_spec in zip(self.reductions, self.reduction_specs):
            value = self.compute_reduction(dirHD_XY, dirHD_YX, reduction_func)
            # merge computation result dict with reduction information dict
            reduction_result = {**reduction_spec.asdict(), **{'value' : value}}
            results.append(reduction_result)
        return results
    
    def _reduction_specs_str(self, style: str = 'str') -> str:
        if style == 'str':
            center = ', '.join((str(s) for s in self.reduction_specs))
        elif style == 'repr':
            center = ', '.join((repr(s) for s in self.reduction_specs))
        else:
            raise ValueError(f'invalid style argument "{style}" must be "str" or "repr"')
        return ''.join(('[', center, ']'))
    
    def _core_kwarg_str_repr(self) -> str:
        attrdict = self.base_hausdorff._get_core_attr_dict()
        # in the context of this class, the reduction of the basal Hausdorff
        # distance computation object is meaningless
        attrdict.pop('reduction')
        core = dict_as_str_repr(attrdict)
        return core
    
    def _as_string(self, reduction_specs_string_style: str) -> str:
        prefix = ''.join((self.__class__.__name__, '('))
        core = (f'reduction_specs={self._reduction_specs_str(style=reduction_specs_string_style)}, '
                f'{self._core_kwarg_str_repr()}, parallelized_reduction='
                f'{self.parallelized_reduction}')
        return ''.join((prefix, core, ')'))
    
    def __str__(self) -> str:
        return self._as_string(reduction_specs_string_style='str')
    
    def __repr__(self) -> str:
        return self._as_string(reduction_specs_string_style='repr')