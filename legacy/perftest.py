import time
import numpy as np
import matplotlib.pyplot as plt



# from metrics import euclidean
from hd2 import *




def make_pointsets(setsize=50**3, pointdim=3, pointdtype=np.int64, verbose=False):
    maxidx = 300
    choices = np.arange(maxidx, dtype=pointdtype)

    pos_x = np.random.default_rng().choice(choices, size=(setsize, pointdim))
    pos_y = np.random.default_rng().choice(choices, size=(setsize, pointdim))

    arrs = [pos_x, pos_y]
    for pos in arrs:
        assert pos.flags['C_CONTIGUOUS'], 'inefficient memory layout'
        if verbose:
            print('\nDIAGNOSTICS')
            print(pos.dtype)
            print(pos.shape)
            print(pos.flags)

    return (pos_x, pos_y)


def benchmark(func, pos_x, pos_y, fkwargs=None):

    # compilation
    mock_x = np.ones(shape=(16, pos_x.shape[1]), dtype=pos_x.dtype)
    mock_y = np.ones(shape=(16, pos_y.shape[1]), dtype=pos_y.dtype)
    if fkwargs:
        assert 'metric' in fkwargs, 'metric callable missing'
        tc_i = time.time()
        _ = func(mock_x, mock_y, **fkwargs)
        tc_f = time.time()
    else:
        tc_i = time.time()
        _ = func(mock_x, mock_y)
        tc_f = time.time()
    
    tc_res = tc_f - tc_i

    # runtime with big payload
    if fkwargs:
        assert 'metric' in fkwargs, 'metric callable missing'
        tr_i = time.time()
        numres = func(pos_x, pos_y, **fkwargs)
        tr_f = time.time()
    else:
        tr_i = time.time()
        numres = func(pos_x, pos_y)
        tr_f = time.time()

    tr_res = tr_f - tr_i

    return {'timings' : {'comptime' : tc_res, 'runtime' : tr_res}, 'result' : numres}

print(f'Runtime estimate: {estimate_runtime(setsize=210000, threads=4)} s')

mt_dirHD = parallelize(dirHD, n_threads=6)
mt_HD = parallelize(HD, n_threads=4)
mt_tracked_dirHD = parallelize_tracked(tracked_dirHD, n_threads=6)

funcs = [tracked_dirHD, mt_tracked_dirHD]

X, Y = make_pointsets(setsize=int(0.5 * 10**5), pointdim=3)

for f in funcs:
    res = benchmark(f, X, Y, fkwargs={'metric' : euclidean})

    runtime = res['timings']['runtime']
    distres, idxres = res['result']

    print(f'Benching function: {f.__name__}')
    print(f'Runtime :: {runtime:.4f} s')
    print(distres[:5])

