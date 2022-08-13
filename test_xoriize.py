from itertools import count
import numpy as np
import matplotlib.pyplot as plt

from xorize import *




def test_volumes() -> tuple:
    spatial_shape = (100, 100, 100)
    edge_length = 30
    centers = [(30, 30, 30), (40, 40, 40)]
    volumes = [np.zeros(spatial_shape, dtype=np.int32)
               for _ in range(2)]
    offset = edge_length // 2
    slices = [
        tuple(slice(axctr-offset, axctr+offset) for axctr in center)
        for center in centers
    ]
    for volume, slc in zip(volumes, slices):
        volume[slc] = 1

    return volumes


volumes = test_volumes()

sumvol = volumes[0] + volumes[1]

a_ex, count = exclude_intersection(volumes[0], volumes[1])

fig, axes = plt.subplots(ncols=2, nrows=2)
axes = axes.flat

ax = axes[0]
ax.imshow(volumes[0][30, ...])

ax = axes[1]
ax.imshow(volumes[1][40, ...])

ax = axes[2]
ax.imshow(sumvol[40, ...])

ax = axes[3]
ax.imshow(a_ex[40, ...])


plt.show()