"""Pixel surface metrics using the vegann-seg-benchmark conventions.

Mean per image; both masks empty -> distances 0/F1 1;
one empty -> image diagonal/F1 0. Foreground is class 1 after mask decoding.
"""
import numpy as np
from scipy import ndimage


def surface_scores(prediction, target, tolerance=2.0):
    def surface(mask):
        return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    p, t = surface(prediction), surface(target)
    if not p.any() and not t.any(): return 0., 0., 1.
    if not p.any() or not t.any():
        diagonal = float(np.hypot(*prediction.shape))
        return diagonal, diagonal, 0.
    dp = ndimage.distance_transform_edt(~t)[p]
    dt = ndimage.distance_transform_edt(~p)[t]
    hd95 = float(max(np.percentile(dp, 95), np.percentile(dt, 95)))
    assd = float(np.concatenate([dp, dt]).mean())
    precision, recall = float(np.mean(dp <= tolerance)), float(np.mean(dt <= tolerance))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return hd95, assd, f1
