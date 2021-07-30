import numpy as np

import hausdorff as hd
from metrics import euclidean
import seghausdorff as shd

# seg_hd_metric = shd.segmentation(hd.hd_metric)

def main():
    arr_1 = np.random.default_rng().integers(0, 2, (10, 10))
    arr_2 = np.random.default_rng().integers(0, 2, (10, 10))

    rawres = hd.hd_metric(arr_1, arr_2, metric=euclidean)

    print(dir(shd))

    print('\n\n\n')

    for name in dir(hd):
        obj = getattr(hd, name)
        print(f'Name: {name}    --  type {type(obj)}')
    # print(dir(hd))


    decres = shd.hd_metric(arr_1, arr_2, metric=euclidean)

    print(f'rawres: {rawres}')
    print(f'segres: {decres}')




if __name__ == '__main__':
    main()