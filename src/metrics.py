"""
Provides metric functions `d(a,b)` for the R^3 space. 
"""
import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, fastmath=True)
def squared_euclidean(a: np.ndarray, b: np.ndarray):
    dsquared = 0
    for i in range(3):
        dsquared += (a[i] - b[i]) ** 2
    return dsquared


@nb.jit(nopython=True, nogil=True, fastmath=True)
def euclidean(a: np.ndarray, b: np.ndarray):
    d = 0
    for i in range(3):
        d += (a[i] - b[i]) ** 2
    return np.sqrt(d)
    