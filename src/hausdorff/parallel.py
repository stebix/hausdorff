import os
import logging
import numpy as np
import concurrent.futures

from typing import Callable, List, Tuple

logger = logging.getLogger('.'.join(('main', __name__)))


def compute_elements_per_worker(total_elements: int, n_workers: int) -> int:
    return int(np.ceil(total_elements / n_workers))


def compute_slices(X: np.ndarray, n_workers: int) -> List[slice]:
    elements_per_worker = compute_elements_per_worker(X.shape[0], n_workers)
    slices = [
        slice(i*elements_per_worker, (i+1)*elements_per_worker)
        for i in range(n_workers)
    ]
    return slices

def hardware_check(n_workers: int):
    """
    Perform a small hardware check and warn user if desired multiprocessing config
    is not sensible concerning the system hardware.
    """
    # maybe more sophisticated in the future
    cpu_count = os.cpu_count()
    if cpu_count < n_workers:
        logger.warning(
            f'Requesting {n_workers} processes on system with {cpu_count} CPU cores. '
            f'Attempting to use more processes than cores may lead to decreased performance'
        )


def parallel_compute_hausdorff(X: np.ndarray, Y: np.ndarray, n_workers: int, *,
                               compute_func: Callable, argtracked: bool = False) -> np.ndarray:
    """
    Compute the (directed) Hausdorff distance for the point sets parallelized across multiple
    worker processes.

    Parameters
    ----------

    X : np.ndarray
        First point set.
    
    Y : np.ndarray
        Second point set.

    n_workers : int
        Number of worker processes.
    
    compute_func : Callable
        The core Hausdorff computation function. Must be a callable with
        the signature `compute_func(X: np.ndarray, Y: np.ndarray)`
    
    argtracked : boolean, optional
        Flag to indicate the behaviour of the `compute_func`. If set to `True`
        it is expected that the index of the minimal distance element of the 
        second point set is returned along with the distance itself.
        Defaults to `False`.

    Returns
    -------

    computation_results : np.ndarray
        If the an argument-tracked implementation is used, the returned array shape
        is `(X.shape[0], 2)` for the minimal distance and the index.
        For the other case it is `(X.shape[0])` for the minimal distance only. 
    """
    if argtracked:
        shape = (X.shape[0], 2)
    else:
        shape = X.shape[0]
    computation_results = np.full(shape, fill_value=np.nan)
    slices = compute_slices(X, n_workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:

        future_to_slice = {
            executor.submit(compute_func, X=X[slc], Y=Y) : slc
            for slc in slices
        }

        for future in concurrent.futures.as_completed(future_to_slice):
            slc = future_to_slice[future]
            try:
                subset_results = future.result()
            except Exception as exc:
                logger.exception(f'computation for slice {slc} failed!', exc_info=exc)
            else:
                computation_results[slc] = subset_results
    return computation_results



def parallel_compute_reduction(dir_XY: np.ndarray, dir_YX: np.ndarray,
                               reduction_func: callable) -> Tuple[np.ndarray]:
    """
    Compute the indicated reduction in parallel for the two directed Hausdorff
    distance results dHD(X,Y) and dHD(Y,X)
    """
    # if directed Hausdorff distance is argtracked we have a 2D array where the
    # first element of a row is the minimal distance and the second element is
    # the corresponding minimal distance element index of the secondary array
    arrays = (dir_XY, dir_YX)
    n_workers = 2
    computation_results = [None, None]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(reduction_func, arrays[i]) : i for i in range(2)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                reduction_result = future.result()
            except Exception as exc:
                logger.exception(f'computation for index {idx} failed!', exc_info=exc)
            else:
                computation_results[idx] = reduction_result
    return tuple(computation_results)