import threading as td
import numpy as np
import numba as nb

from functools import wraps
from queue import Queue
from math import sqrt, ceil
from typing import Callable, Tuple

@nb.jit(nopython=True, fastmath=True, nogil=True)
def euclidean(X, Y):
    distance = 0
    for i in range(X.size):
        distance += np.power(X[i] - Y[i], 2)
    return np.sqrt(distance)


@nb.jit(nopython=True, fastmath=True, nogil=True)
def dirHD(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> float:
    assert X.ndim == Y.ndim == 2, 'point coordinate set required'
    assert X.shape[1] == Y.shape[1], 'identical point dimensionality required'
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    maxdist = np.float64(0.0)
    mindist = np.float64(0.0)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            if dist < mindist:
                mindist = dist
        if maxdist < mindist:
            maxdist = mindist
    
    return maxdist


def parallelize(func: Callable, n_threads: int = 2) -> Callable:

    def worker(func, args, kwargs, queue):
        result = func(*args, **kwargs)
        queue.put(result)
        return None

    @wraps(func)
    def multithreaded_func(*args, **kwargs):
        # pull out set arguments
        X = args[0]
        Y = args[1]
        card_X = X.shape[0]
        assert card_X > 2 * n_threads, 'X set size too small for parallelization'

        # threading setup
        threads = []
        # buffer for all thread worker results
        q = Queue(maxsize=n_threads)
        maxdistances = []

        # split X into n_threads disjunct partitions and compute
        chunksize = ceil(card_X / n_threads)
        for i in range(n_threads):
            slc = slice(i*chunksize, (i+1)*chunksize)
            X_subset = X[slc]
            
            threads.append(
                td.Thread(target=worker, args=(func, (X_subset, Y), kwargs, q))
            )
        
        for t in threads:
            t.start()
        # collect results before joining to avoid queue-join lockup
        while len(maxdistances) < n_threads:
            maxdistances.append(q.get(block=True))
        for t in threads:
            t.join()
        
        return np.max(maxdistances)
    
    return multithreaded_func



def HD(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> float:
    maxdist_XY = dirHD(X, Y, metric=metric)
    maxdist_YX = dirHD(Y, X, metric=metric)
    return max((maxdist_XY, maxdist_YX))


@nb.jit(nopython=True, fastmath=True, nogil=True)
def tracked_dirHD(X: np.ndarray, Y: np.ndarray, *, metric: Callable) -> Tuple:
    assert X.ndim == Y.ndim == 2, 'point coordinate set required'
    assert X.shape[1] == Y.shape[1], 'identical point dimensionality required'
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    mindist = 0.0
    distances = np.full(shape=card_X, fill_value=np.nan, dtype=np.float64)
    indices = np.full(shape=card_X, fill_value=-1, dtype=np.int64)

    for i in range(card_X):
        mindist = np.inf
        minidx = -1
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            if dist < mindist:
                mindist = dist
                minidx = j    
        distances[i] = mindist
        indices[i] = minidx
        
    return (distances, indices)



def parallelize_tracked(func: Callable, n_threads: int = 2) -> Callable:

    def worker(func, args, kwargs, queue, slc):
        result = func(*args, **kwargs)
        queue.put((slc, result))
        return None

    def multithreaded_func(*args, **kwargs):
        # pull out set arguments
        X = args[0]
        Y = args[1]
        card_X = X.shape[0]
        assert card_X > 2 * n_threads, 'X set size too small for parallelization'

        # threading setup
        threads = []
        # buffer for all thread worker results
        q = Queue(maxsize=n_threads)
        distances = np.full(card_X, np.inf, dtype=np.float64)
        indices = np.full(card_X, -1, np.int64)
        resultbuffer = []

        # split X into n_threads disjunct partitions and compute
        chunksize = ceil(card_X / n_threads)
        for i in range(n_threads):
            slc = slice(i*chunksize, (i+1)*chunksize)
            
            threads.append(
                td.Thread(target=worker, args=(func, (X[slc], Y), kwargs, q, slc))
            )
        
        for t in threads:
            t.start()
        # collect results before joining to avoid queue-join lockup
        while len(resultbuffer) < n_threads:
            resultbuffer.append(q.get(block=True))
        for t in threads:
            t.join()

        # merge results from multiple threads
        for partial_result in resultbuffer:
            slc, (part_distcs, part_idcs) = partial_result
            distances[slc] = part_distcs
            indices[slc] = part_idcs
        
        return (distances, indices)
    
    return multithreaded_func



def estimate_runtime(setsize, threads=1):
    elements = setsize ** 2
    rtime_per_elem_estim = 11 / ((0.5 * 10 ** 5) ** 2)
    return rtime_per_elem_estim * elements / threads


