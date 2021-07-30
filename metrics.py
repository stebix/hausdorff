import numpy as np
import numba as nb

from math import sqrt


@nb.jit(nopython=True, fastmath=True, parallel=False)
def euclidean(X: np.ndarray, Y: np.ndarray) -> float:
    X = X.flatten()
    Y = Y.flatten()
    assert X.size == Y.size, 'Identical shape required'
    distance = 0.0
    for i in nb.prange(X.size):
        distance += np.power(X[i] - Y[i], 2)

    return np.sqrt(distance)


@nb.jit(nopython=True, fastmath=True, parallel=False)
def mineuc(X, Y):
    distance = 0.0
    for i in nb.prange(X.size):
        distance += np.power(X[i] - Y[i], 2)

    return np.sqrt(distance)



@nb.jit(nopython=True, fastmath=True, parallel=False)
def euclidean_mimp(X: np.ndarray, Y: np.ndarray) -> float:
    X = X.flatten()
    Y = Y.flatten()
    assert X.size == Y.size, 'Identical shape required'
    distance = 0.0
    for i in nb.prange(X.size):
        distance += (X[i] - Y[i]) ** 2

    return sqrt(distance)