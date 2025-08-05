import numpy as np
import matplotlib.pyplot as plt
import pytest

from hausdorff.intersectiontools import mask_unique_true, postpad
from hausdorff.volumehelpers import create_overlapping_cubes


def create_multiplot(ncols: int, nrows: int = 1) -> tuple:
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    axes = axes.flat
    return (fig, axes)

class Test_postpad:

    @pytest.mark.parametrize('shape', [(33, 2), (50,), (20, 2)])
    def test_is_noop_for_padwidth_zero(self, shape):
        array = np.random.default_rng().normal(size=shape)
        result = postpad(array, pad_width=0)
        assert np.array_equal(array, result), 'postpad modified array despite pad_width = 0'
    
    @pytest.mark.parametrize('shape', [(33,), (20,)])
    def test_pads_correct_length_on_1D_arrays(self, shape):
        pad_width = 25
        array = np.ones(shape)
        result = postpad(array, pad_width=pad_width)
        expected_shape = (shape[0] + pad_width,)
        assert result.shape == expected_shape
        assert np.array_equal(result[-pad_width:], np.zeros((pad_width,)))
    
    @pytest.mark.parametrize('shape', [(33, 2), (20, 2)])
    def test_pads_only_first_axis_on_2D_arrays(self, shape):
        pad_width = 25
        array = np.ones(shape)
        result = postpad(array, pad_width=pad_width)
        expected_shape = (shape[0] + pad_width, shape[1])
        assert result.shape == expected_shape
        assert np.array_equal(result[-pad_width:, :], np.zeros((pad_width, shape[1])))


class Test_mask_unique_true:

    def test_basic_functionality(self):
        va, vb = create_overlapping_cubes(as_bool=True)
        intersection = np.logical_and(va, vb)
        result, vb = mask_unique_true(va, vb, return_secondary=True,
                                    return_intersection_count=False)
        expected = np.logical_and(va, np.logical_not(intersection))
        assert np.array_equal(result, expected)
    
    def test_mask_and_inverted_intersection_gives_original(self):
        va, vb = create_overlapping_cubes(as_bool=True)
        intersection = np.logical_and(va, vb)
        result = mask_unique_true(va, vb, return_secondary=False,
                                  return_intersection_count=False)
        reunion = np.logical_or(result, intersection)
        assert np.array_equal(va, reunion)

    def test_intersection_count(self):
        va, vb = create_overlapping_cubes(as_bool=True)
        intersection = np.logical_and(va, vb)
        expected_count = np.sum(intersection)
        _, vb, result_count = mask_unique_true(va, vb, return_secondary=True,
                                               return_intersection_count=True)
        assert expected_count == result_count
    
    def test_returntuple_absorption(self):
            va, vb = create_overlapping_cubes(as_bool=True)
            intersection = np.logical_and(va, vb)
            expected = np.logical_and(va, np.logical_not(intersection))
            (*args, count) = mask_unique_true(va, vb)
            assert np.array_equal(args[0], expected), 'returntuple index 0 not identical to expected result'
            assert np.array_equal(args[1], vb), 'returntuple index 1 not identical to second input array'
            assert np.isclose(intersection.sum(), count), 'intersection count not as expected'
