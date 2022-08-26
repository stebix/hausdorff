"""
Raw Hausdorff distance and metric calculation implementation for point sets `X` and `Y`.
"""

import multiprocessing as mp
import numpy as np
import numba as nb

from typing import List, Tuple, Dict, Optional, Callable, Union, Sequence


@nb.jit(nopython=True, fastmath=True, parallel=True)
def pairwise_distances(X: np.ndarray, Y: np.ndarray, *,
                       metric: Callable) -> np.ndarray:
    
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.zeros(card_X*card_Y, dtype=np.float32)
    for i in range(card_X):
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            distances[i*j] = dist
    return distances


@nb.jit(nopython=True, fastmath=True, parallel=False)
def dir_hd_distcs(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> np.ndarray:
    """
    Compute the directed Hausdorff distances between the sets `X` and `Y`.
    Return an array of distances:
    For every element x € X, the minimum distance d_min = metric(x, y_min)
    with y_min € Y is provided.
    Hint: X and Y should be provided as positional arguments only.

    Parameters
    ----------

    X: numpy.ndarray
        First set for dirH(X, Y)

    Y: numpy.ndarray
        Second set for dirH(X, Y)
    
    metric: Callable
        The metric function used to calculate
        dist = metric(x, y) with x € X and y € Y.

    Returns
    -------

    distances: numpy.ndarray
        The 1D array of minimal distances with a length
        of X.shape[0].
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.zeros(card_X, dtype=np.float32)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            if dist < mindist:
                mindist = dist
        distances[i] = mindist
    return distances


@nb.jit(nopython=True, fastmath=True)
def dir_hd_dist(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> float:
    """
    Compute the directed Hausdorff distance.
    Hint: X and Y should be provided as positional arguments only.

    Parameters
    ----------

    X: numpy.ndarray
        First set for dirH(X, Y)

    Y: numpy.ndarray
        Second set for dirH(X, Y)
    
    metric: Callable
        The metric function used to calculate
        dist = metric(x, y) with x € X and y € Y.

    Returns
    -------

    hd_distance: float
        The directed Hausdorff distance dH(X, Y).
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    maxdist = 0.0
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            if dist < mindist:
                mindist = dist
        if maxdist < mindist:
            maxdist = mindist
    return maxdist



@nb.jit(nopython=True, fastmath=True, parallel=False)
def argtr_dir_hd_distcs(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> Tuple[np.ndarray]:
    """
    Computes the argument-tracked directed Hausdorff distances between `X` and `Y`.
    This means that the distance and the
    corresponding index of the element of `Y` that gives the minimal distance
    is calculated:
    dist_i = metric(x_i, y*_j)   where y*_j minimizes the distances over all y € Y.

    Hint: X and Y should be provided as positional arguments only.


    Parameters
    ----------

    X: numpy.ndarray
        First set for dirH(X, Y)

    Y: numpy.ndarray
        Second set for dirH(X, Y)
    
    metric: Callable
        The metric function used to calculate
        dist = metric(x, y) with x € X and y € Y.

    Returns
    -------

    (distances, indices): 2-tuple of numpy.ndarray
        The tuple of distances and corresponding indices of elements
        of Y that minimize the distance. 
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.zeros((card_X), dtype=np.float32)
    indices = np.full(card_X, np.nan, dtype=np.int32)
    for i in range(card_X):
        mindist = np.inf
        minidx = np.nan
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            if dist < mindist:
                mindist = dist
                minidx = j
        distances[i] = mindist
        indices[i] = minidx
    assert np.all(~np.isnan(indices)), 'WTF'
    return (distances, indices)



@nb.jit(nopython=True, fastmath=True)
def hd_distcs(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> np.ndarray:
    """
    Compute the Hausdorff distances between `X` and `Y` as well as the mirror
    case `Y` and `X`.
    Hint: X and Y should be provided as positional arguments only.

    Parameters
    ----------

    X: numpy.ndarray
        First set.

    Y: numpy.ndarray
        Second set.
    
    metric: Callable
        The metric function used to calculate
        dist = metric(x, y) with x € X and y € Y.
    
    Returns
    -------

    (hdists_XY, hdists_YX) : 2-tuple of np.ndarray
        Tuple of both directed Hausdorff distances dH(X, Y)
        and dH(Y, X). 
    """
    hdists_XY = dir_hd_distcs(X, Y, metric=metric)
    hdists_YX = dir_hd_distcs(Y, X, metric=metric)
    return (hdists_XY, hdists_YX)


def hd_distcs_pll(X: np.ndarray, Y: np.ndarray, *, metric: Callable):
    """
    Parallelized Hausdorff distances calculation of (dH(X, Y), dH(Y, X))
    distributed across two processes.
    Hint: X and Y should be provided as positional arguments only.

    Parameters
    ----------

    X: numpy.ndarray
        First set.

    Y: numpy.ndarray
        Second set.
    
    metric: Callable
        The metric function used to calculate
        dist = metric(x, y) with x € X and y € Y.
    
    Returns
    -------

    (hdists_XY, hdists_YX) : 2-tuple of np.ndarray
        Tuple of both directed Hausdorff distances dH(X, Y)
        and dH(Y, X). 

    """
    queue = mp.Queue(3)
    worker_XY = mp.Process(target=_worker,
                           args=(dir_hd_distcs, (X, Y), {'metric' : metric}, queue, 0))
    worker_YX = mp.Process(target=_worker,
                           args=(dir_hd_distcs, (Y, X), {'metric' : metric}, queue, 1))

    results = []

    workers = [worker_XY, worker_YX]
    for w in workers:
        w.start()

    # collect results from queue BEFORE joining to circumvent eternal .join()
    # hang ... be aware that this loop runs forever if a worker fucks up
    while len(results) < 2:
        results.append(queue.get(block=True))

    for w in workers:
        w.join()
    results.sort(key=lambda tpl : tpl[0])
    return tuple(r[1] for r in results)


def _worker(func: Callable, fargs: Tuple, fkwargs: dict, queue: mp.Queue, pos: int) -> None:
    """
    Worker template for multiprocessing implementation in `hd_distcs_pll`.
    The callable `func` is evaluated with `fargs` and `fkwargs`, its results are placed in
    the provided queue as a tuple with the `pos` integer as a canonical
    ordering index.
    """
    result = func(*fargs, **fkwargs)
    queue.put((pos, result))


def avg_hd_dist(X: np.ndarray, Y: np.ndarray, *, metric: Callable, parallel=True) -> float:
    """
    Hint: X and Y should be provided as positional arguments only.
    """
    if parallel:
        dists = hd_distcs_pll(X, Y, metric=metric)
    else:
        dists = hd_distcs(X, Y, metric=metric)
    
    # accumulate the mean of the directed Hausdorff distances
    dir_mean_acc = 0.0
    for dir_dists in dists:
        dir_mean_acc += np.mean(dir_dists)
    # average again
    return dir_mean_acc / 2


def hd_metric(X: np.ndarray, Y: np.ndarray, *, metric: Callable, parallel=True) -> float:
    """
    Compute the Hausdorff metric between the sets `X` and `Y`.
    Hint: X and Y should be provided as positional arguments only.
    """
    if parallel:
        return np.max(np.concatenate(hd_distcs_pll(X, Y, metric=metric), axis=0))
    else:
        return np.max(np.concatenate(hd_distcs(X, Y, metric=metric), axis=0))




def percentile_hd_metric(X: np.ndarray, Y: np.ndarray, *, metric: Callable,
                         q: Union[float, Sequence[float]],
                         **kwargs) -> float:
    """
    Convenience function to compute q-th percentile of Hausdorff distances between
    `X` and `Y`.
    """
    distances = np.concatenate(hd_distcs(X, Y, metric=metric), axis=0)
    return np.percentile(distances, q=q, **kwargs)



