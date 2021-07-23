import numpy as np
import matplotlib.pyplot as plt

import pytest

from hausdorff import *
from metrics import *


"""
pointset_1 = np.random.default_rng().normal(0, 1, size=(1000, 3))
pointset_2 = np.random.default_rng().normal(0, 1, size=(1000, 3))


pdist = pairwise_distances(pointset_1, pointset_2, euclidean)
hdist = hausdorff_metric(pointset_1, pointset_2, euclidean)

hd95 = hausdorff_percentile_distance(pointset_1, pointset_2, euclidean, 0.95)

# print(np.max(hdist))
print(hdist)
print(f'HD95: {hd95}')
"""

test_pred = np.zeros((10, 10), dtype=np.int32)
test_lbl = np.zeros((10, 10), dtype=np.int32)

# prediction positive class coordinates
pred_pos_coords = [(3,3), (4, 3), (3, 4), (4, 4)]
for poscord in pred_pos_coords:
    test_pred[poscord] = 1

# label positive class coordinates
label_pos_coords = [(4, 4), (5, 4),(4, 5), (5, 5), (6, 6)]
for poscord in label_pos_coords:
    test_lbl[poscord] = 1

trafo_pred, trafo_lbl = prepare_segmentation(test_pred, test_lbl)

hd = seg_hausdorff_distances(test_pred, test_lbl, metric=euclidean)
print(hd)

hdm = hausdorff_metric(trafo_pred, trafo_lbl, euclidean)
print(f'Hausdorff Metric: {hdm:.2f}')

pl_dhd, pl_idxs = tracked_directed_hd_distances(trafo_pred, trafo_lbl, euclidean)
lp_dhd, lp_idxs = tracked_directed_hd_distances(trafo_lbl, trafo_pred, euclidean)


fig, axes = plt.subplots(ncols=3)
axes = axes.flat

fig.suptitle(f'Hausdorff Metric: {hdm:.2f}')

ax = axes[0]
ax.set_title('Prediction')
ax.imshow(test_pred)


ax = axes[1]
ax.set_title('Label')
ax.imshow(test_lbl)

ax = axes[2]
ax.set_title('Overlay')
cumul = test_pred + test_lbl
img = ax.imshow(cumul, cmap='Set1')


for pt, idx in zip(trafo_pred, pl_idxs):
    start = tuple(pt)
    stop = tuple(trafo_lbl[idx])
    ax.annotate('', xy=stop, xytext=start,
                arrowprops=dict(arrowstyle='->', color='g'))


for pt, idx in zip(trafo_lbl, lp_idxs):
    start = tuple(pt)
    stop = tuple(trafo_pred[idx])
    ax.annotate('', xy=stop, xytext=start,
                arrowprops=dict(arrowstyle='->', color='r'),
                )



plt.show()