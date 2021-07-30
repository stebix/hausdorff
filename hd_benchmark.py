import time
import pathlib

from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

import hausdorff as hd
from metrics import euclidean, euclidean_mimp



def bench_hd_func(func: Callable, arrshape: Tuple, metric: Callable):
    """
    Benchmark a callable Hausdorff function `func` with the expected
    signature `func(X, Y, metric)` with a given array shape.
    Returns a dictionary with the compilation time and the runtime.
    """
    # 'real' benchmark data 
    coordX = np.argwhere(np.random.default_rng().integers(0, 2, size=arrshape))
    coordY = np.argwhere(np.random.default_rng().integers(0, 2, size=arrshape))
    # 'mock' data for compilation run
    mockA = np.random.default_rng().integers(0, 10, size=(10, 2))
    mockB = np.random.default_rng().integers(0, 10, size=(10, 2))
    # compilation
    tc_i = time.time()
    _ = func(mockA, mockB, metric=metric)
    tc_f = time.time()
    tc = tc_f - tc_i
    # actual payload run
    tp_i = time.time()
    res = func(coordX, coordY, metric=metric)
    tp_f = time.time()
    tp = tp_f - tp_i
    return {'comptime' : tc, 'runtime' : tp}




def bench_dir_hd_distcs(arrshape=(50, 50), metric=euclidean):
    """
    Benchmark the function `dir_hd_distcs`
    """
    return bench_hd_func(hd.dir_hd_distcs, arrshape, metric)


def bench_dir_hd_dist(arrshape=(50, 50), metric=euclidean):
    return bench_hd_func(hd.dir_hd_dist, arrshape, metric)


def bench_hd_distcs(arrshape=(50, 50), metric=euclidean):
    return bench_hd_func(hd.hd_distcs, arrshape, metric)

def bench_p_hd_distcs(arrshape=(50, 50), metric=euclidean):
    coordX = np.argwhere(np.random.default_rng().integers(0, 2, size=arrshape))
    coordY = np.argwhere(np.random.default_rng().integers(0, 2, size=arrshape))

    t_i = time.time()
    result = hd.hd_distcs_pll(coordX, coordY, metric)
    t_f = time.time()

    return {'comptime' : None, 'runtime' : t_f - t_i}



if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()

    arrshape = (100, 100)
    print(bench_hd_func(hd.argtr_dir_hd_distcs, arrshape=(50, 50), metric=euclidean))
    print(bench_hd_distcs(arrshape))
    print(bench_p_hd_distcs(arrshape))

    # print(bench_dir_hd_distcs(arrshape=arrshape))
    # print(bench_dir_hd_distcs(arrshape=arrshape, metric=euclidean_mimp))

    # print(bench_dir_hd_dist(arrshape=arrshape))
    # print(bench_dir_hd_dist(arrshape=arrshape, metric=euclidean_mimp))



