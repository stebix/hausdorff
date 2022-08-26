import numpy as np
import numba  as nb

from typing import Callable


        

@nb.jit(nopython=True, fastmath=True, nogil=True)
def directed_hausdorff_distances(X: np.ndarray, Y: np.ndarray, *,
                                 metric: Callable) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and a supplied metric.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.empty(card_X)
    for i in range(card_X):
        mindist = np.inf
        for j in range(card_Y):
            dist = metric(X[i], Y[j])
            if dist < mindist:
                mindist = dist
        distances[i] = mindist
    return distances


@nb.jit(nopython=True, fastmath=True)
def directed_hausdorff_distances_argtracked(X: np.ndarray, Y: np.ndarray, *,
                                            metric: Callable) -> np.ndarray:
    """
    Compute the array of directed Hausdorff distances for two point sets
    X and Y in R^3 and a supplied metric. The minimal distance index of
    the second set is tracked during the computation.
    """
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    indextracked_distances = np.zeros((card_X, 2))
    for i in range(card_X):
        mindist = np.inf
        minidx = np.nan
        for j in range(card_Y):
            dist = metric(X[i], Y[j])
            if dist < mindist:
                mindist = dist
                minidx = j
        indextracked_distances[i, 0] = mindist
        indextracked_distances[i, 1] = minidx
    return indextracked_distances