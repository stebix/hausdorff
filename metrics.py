import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True, parallel=False)
def euclidean(X: np.ndarray, Y: np.ndarray) -> float:
    X = X.flatten()
    Y = Y.flatten()
    assert X.size == Y.size, 'Identical shape required'
    distance = 0.0
    for i in range(X.size):
        distance += np.power(X[i] - Y[i], 2)

    return np.sqrt(distance)