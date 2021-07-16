import numpy as np
import numba as nb

from typing import List, Tuple, Dict, Optional, Callable, Union, Sequence



def is_binary(arr: np.array) -> bool:
    """
    Check if all array elements are approximately binary,
    i.e. elem € {0, 1} within float tolerance.

    Parameters
    ----------

    arr : numpy.ndarray
        The array to test.

    Returns
    -------

    is_binary : bool
        Boolean flag indicating the binary property.
    """
    bool_arr = np.logical_or(
        np.isclose(arr, 0), np.isclose(arr, 1)
    )
    if np.all(bool_arr):
        return True
    else:
        return False


def is_idshape(X: np.ndarray, Y: np.ndarray) -> bool:
    if X.shape == Y.shape:
        return True
    else:
        return False



def prepare_segmentation(pred: np.ndarray,
                         label: np.ndarray,
                         exclude_intersection: bool = False
                         ) -> Tuple[np.ndarray]:
    """
    Transform a binary semantic segmentation pair < prediction, label> towards
    two coordinate vector arrays.
    """
    assert is_binary(pred), 'Prediction volume must be binary'
    assert is_binary(label), 'Label volume must be binary'
    assert is_idshape(pred, label), 'Mismatching prediction and label volume shape'
    # recast as boolean arrays
    pred = np.isclose(pred, 1.0)
    label = np.isclose(label, 1.0)
    label_coords = np.argwhere(label)
    if exclude_intersection:
        xor_mask = np.logical_xor(pred, label)
        pred_coords = np.argwhere(xor_mask)
    else:
        pred_coords = np.argwhere(pred)

    return (pred_coords, label_coords)



@nb.jit(nopython=True, fastmath=True, parallel=True)
def pairwise_distances(X: np.ndarray, Y: np.ndarray,
                       metric: Callable) -> np.ndarray:
    
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.zeros(card_X*card_Y, dtype=np.float32)
    for i in range(card_X):
        for j in range(card_Y):
            dist = metric(X[i, ...], Y[j, ...])
            distances[i*j] = dist
    return distances


@nb.jit(nopython=True, fastmath=True, parallel=True)
def directed_hausdorff_distances(X: np.ndarray, Y: np.ndarray,
                                 metric: Callable
                                 ) -> np.ndarray:
    
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


# @nb.jit(nopython=True, fastmath=True, parallel=True)
def tracked_directed_hd_distances(X: np.ndarray, Y: np.ndarray,
                                  metric: Callable
                                  ) -> np.ndarray:
    
    card_X = X.shape[0]
    card_Y = Y.shape[0]
    distances = np.zeros((card_X), dtype=np.float32)
    indices = np.full(card_X, np.nan, dtype=np.int8)
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
def hausdorff_distances(X: np.ndarray, Y: np.ndarray,
                        metric: Callable) -> np.ndarray:

    hdists_XY = directed_hausdorff_distances(X, Y, metric)
    hdists_YX = directed_hausdorff_distances(Y, X, metric)
    return (hdists_XY, hdists_YX)



def hausdorff_percentile_distance(X: np.ndarray, Y: np.ndarray,
                                  metric: Callable,
                                  q: Union[float, Sequence[float]],
                                  **kwargs) -> float:
    distances = np.concatenate(hausdorff_distances(X, Y, metric), axis=0)
    return np.percentile(distances, q=q, **kwargs)



@nb.jit(nopython=True, fastmath=True, parallel=True)
def hausdorff_metric(X: np.ndarray, Y:np.ndarray, metric: Callable) -> float:
    """
    Compute the Hausdorff metric between the sets `X` and `Y`.
    """
    hd_XY = np.max(directed_hausdorff_distances(X, Y, metric))
    hd_YX = np.max(directed_hausdorff_distances(Y, X, metric))
    return max((hd_XY, hd_YX))


def seg_hausdorff_distances(pred: np.ndarray,
                            label: np.ndarray,
                            metric: Callable,
                            exclude_intersection: bool = False,
                            voxelspacing: Optional[float] = None):

    pred_coords, label_coords = prepare_segmentation(pred, label,
                                                     exclude_intersection)
    hdists = hausdorff_distances(pred_coords, label_coords, metric)
    if voxelspacing:
        return [dir_hdist * voxelspacing for dir_hdist in hdists]
    else:
        return hdists