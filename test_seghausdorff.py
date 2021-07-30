import numpy as np
import pytest

from types import FunctionType

import seghausdorff as shd
from metrics import euclidean




class Test_transform_data:

    def test_changes_dtype(self):
        test_arr = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
        result = shd.transform_data(test_arr, dtype=np.float32)
        assert result.dtype == np.float32

    def test_returns_singleton_on_single_input(self):
        test_arr = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
        result = shd.transform_data(test_arr, dtype=np.float32)
        assert isinstance(result, np.ndarray)

    def test_return_tuple_of_arrays_on_multiple_inputs(self):
        test_arr_1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
        test_arr_2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int32)
        results = shd.transform_data(test_arr_1, test_arr_2)
        assert isinstance(results, tuple), 'Should return a tuple'
        assert all(isinstance(a, np.ndarray) for a in results), 'Elements should be ndarray'



class Test_preprocess:

    def test_fail_on_mismatching_shapes(self):
        arr1 = np.arange(25).reshape(5, 5)
        arr2 = 3 * np.arange(25).reshape(5, 5)
        arr3 = np.arange(36).reshape(6, 6)
        with pytest.raises(AssertionError):
            result = shd.preprocess(arr1, arr2, arr3)

    
    def test_produces_coord_arrs_correct_shape(self):
        arr1 = np.random.default_rng().integers(0, 2, (5, 5))
        result1 = shd.preprocess(arr1)
        assert result1.shape == (np.sum(arr1), 2)
        arr2 = np.random.default_rng().integers(0, 2, (5, 5, 5))
        result2 = shd.preprocess(arr2)
        assert result2.shape == (np.sum(arr2), 3)


class Test_argtr_dir_hd_distcs:

    def test_successful_import(self):
        assert isinstance(shd.argtr_dir_hd_distcs, FunctionType)

    def test_integration_visual(self):
        """
        Large & more complex integration test with required
        user-interaction trough plot inspection.
        """
        import matplotlib.pyplot as plt
        # data setup
        prediction = np.zeros((20, 20), dtype=np.int32)
        label = np.zeros((20, 20), dtype=np.int32)
        # foreground pixels
        pred_fg = [(4, 4), (4, 5), (5, 5), (5, 4), (18, 14)]
        label_fg = [(15, 15), (16, 15), (15, 16), (16, 16), (5, 10)]
        for pfg in pred_fg:
            prediction[pfg] = 1
        for lfg in label_fg:
            label[lfg] = 1
        distances, indices = shd.argtr_dir_hd_distcs(
            prediction, label, metric=euclidean
        )

        prediction_coords = np.argwhere(prediction)
        label_coords = np.argwhere(label)

        
        vizarr = prediction + 2 * label
        fig, ax = plt.subplots()
        ax.matshow(vizarr)

        # draw the arrows, reverse for ij <-> xy
        for prc, idx in zip(prediction_coords, indices):
            print(prc)
            ax.annotate(
                '', xytext=reversed(tuple(prc)),
                xy=reversed(tuple(label_coords[idx, :])),
                arrowprops={'arrowstyle' : '->'}, xycoords='data'
            )
        


        plt.show()
        



