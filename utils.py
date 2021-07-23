import numpy as np

"""
Various utility functions for the `hausdorff` package.
"""


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