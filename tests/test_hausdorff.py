import numpy as np
import pytest
import matplotlib.pyplot as plt

from hausdorff import Hausdorff, DirectedHausdorff
from hausdorff.volumehelpers import create_overlapping_cubes, create_simple_testvolume


class Test_DirectedHausdorff:

    @pytest.mark.parametrize('parallelized,n_workers', [(False, 0), (True, 2), (True, 3)])
    @pytest.mark.parametrize('metric,postprocess', [('euclidean', 'none'), ('squared_euclidean', 'sqrt')])
    def test_euclidean_distance_simple(self, metric, postprocess, parallelized, n_workers):
        hausdorff = DirectedHausdorff(
            reduction='canonical', remove_intersection=False,
            metric=metric, postprocess=postprocess,
            parallelized=parallelized, n_workers=n_workers
        )
        shape = (50, 50, 50)
        pos_a = (25, 10, 10)
        pos_b = (25, 40, 40)
        va, vb = create_simple_testvolume(shape, pos_a, pos_b)
        # recast as numpy.ndarray
        pos_a, pos_b = (np.array(pos) for pos in (pos_a, pos_b))
        difference = pos_a - pos_b
        expected_distance = np.linalg.norm(difference)
        result = hausdorff.compute(va, vb)
        assert np.isclose(expected_distance, result)


class Test_Hausdorff:

    @pytest.mark.parametrize('remove_intersection', [True, False])
    def test_with_shifted_cubes(self, remove_intersection):
        hausdorff = Hausdorff(
            reduction='average', remove_intersection=remove_intersection,
            metric='euclidean', postprocess='none',
            parallelized=False, n_workers=0
        )
        va, vb = create_overlapping_cubes()
        result = hausdorff.compute(va, vb)
        print(result)


def test_plot_overlap():
    va, vb = create_overlapping_cubes()
    axis_0_idx = 30
    intersection = np.logical_and(va, vb)
    volsum = va + vb

    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flat
    
    ax = axes[0]
    ax.set_title('cube a')
    ax.imshow(va[axis_0_idx, ...])

    ax = axes[1]
    ax.set_title('cube b')
    ax.imshow(vb[axis_0_idx, ...])

    ax = axes[2]
    ax.set_title('intersection')
    ax.imshow(intersection[axis_0_idx, ...])

    ax = axes[3]
    ax.set_title('sum')
    ax.imshow(volsum[axis_0_idx, ...])

    plt.show()


def test():
    hausdorff = Hausdorff(
        reduction='canonical', remove_intersection=False,
        metric='squared_euclidean', postprocess='sqrt',
        parallelized=True, n_workers=4
    )
    shape = (50, 50, 50)
    pos_a = (25, 10, 10)
    pos_b = (25, 40, 40)
    va, vb = create_simple_testvolume(shape, pos_a, pos_b)

    result = hausdorff.compute(va, vb)

    fig, axes = plt.subplots(ncols=2)
    fig.suptitle(f'HD = {result}')
    axes = axes.flat
    ax = axes[0]
    ax.imshow(va[25, ...])
    ax = axes[1]
    ax.imshow(vb[25, ...])
    plt.show()
    

