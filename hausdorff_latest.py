import time
import numpy as np
import numba as nb

from typing import Sequence
from collections import defaultdict

import dirhd 


@nb.jit(nopython=True, nogil=True, fastmath=True)
def squared_euclidean(a: np.ndarray, b: np.ndarray):
    d = 0
    for i in range(3):
        d += (a[i] - b[i]) ** 2
    return d
        

@nb.jit(nopython=True, fastmath=True, nogil=True)
def directed_hausdorff_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and the Euclidean metric.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    squared_distances = np.zeros(card_X)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = 0
            for k in range(3):
                dist += (X[i, k] - Y[j, k]) ** 2
            if dist < mindist:
                mindist = dist
        squared_distances[i] = mindist
    return squared_distances


@nb.jit(nopython=True, fastmath=True)
def directed_hausdorff_distances_separate(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and the Euclidean metric.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    squared_distances = np.zeros(card_X)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = squared_euclidean(X[i], Y[j])
            if dist < mindist:
                mindist = dist
        squared_distances[i] = mindist
    return squared_distances


@nb.jit(nopython=True, fastmath=True)
def directed_hausdorff_distances_separate_argtracked(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and the Euclidean metric.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    indextracked_squared_distances = np.zeros((card_X, 1))
    for i in range(card_X):
        mindist = np.inf
        minidx = np.nan
        for j in range(card_Y):
            dist = squared_euclidean(X[i], Y[j])
            if dist < mindist:
                mindist = dist
                minidx = j
        indextracked_squared_distances[i, 0] = mindist
        indextracked_squared_distances[i, 1] = minidx
    return indextracked_squared_distances



@nb.jit((nb.int32[:, ::1], nb.int32[:, ::1]), nopython=True, fastmath=True)
def directed_hausdorff_distances_integer(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and the Euclidean metric.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    squared_distances = np.zeros(card_X, dtype=np.int32)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = 0
            for k in range(3):
                dist += (X[i, k] - Y[j, k]) ** 2
            if dist < mindist:
                mindist = dist
        squared_distances[i] = mindist
    return squared_distances

def format_runtimes(runtimes: Sequence[float]) -> str:
    """
    Pretty-format a list of runtimes for a code snippet as a nicely readable
    string.
    """
    mean, min_, max_ = np.mean(runtimes), np.min(runtimes), np.max(runtimes)
    N = len(runtimes)
    return f'Best of N = {N}: {min_:.3f}, AVG = {mean:.3f}'


if __name__ == '__main__':

    SEED = 3141
    np.random.RandomState(SEED)

    HD_funcs = {
        'standard' : directed_hausdorff_distances,
        'separate metric function' : directed_hausdorff_distances_separate,
        'separate +  argtracked metric function' : directed_hausdorff_distances_separate_argtracked,
        'integer specialized' : directed_hausdorff_distances_integer,
        'cython' : dirhd.directed_hausdorff_distances_integer
    }

    DTYPE = np.int32

    # warmup via JIT compilation, careful with matching data types
    warmup_X = np.random.randint(0, 100, size=(5, 3), dtype=DTYPE)
    warmup_Y = np.random.randint(0, 100, size=(5, 3), dtype=DTYPE)
    for func in HD_funcs.values():
        _ = func(warmup_X, warmup_Y)


    # actual data setup
    elements = int(2e4)
    test_X = np.random.randint(0, 100, size=(elements, 3), dtype=DTYPE)
    test_Y = np.random.randint(0, 100, size=(elements, 3), dtype=DTYPE)

    # benchamrk settings
    N_REPEATS = 5
    benchmark_results = defaultdict(list)

    test_result = None

    for name, func in HD_funcs.items():
    
        for _ in range(N_REPEATS):
            start = time.perf_counter()
            result = func(test_X, test_Y)
            stop = time.perf_counter()
            runtime = stop - start
            benchmark_results[name].append(runtime)

        print(f'Computation with implementation "{name}":')
        print(format_runtimes(benchmark_results[name]))
        print('')

    #print(f'Result is\n{result}')



