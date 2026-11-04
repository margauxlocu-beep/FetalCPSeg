import os
import numpy as np


class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_score(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    p = (pred == class_id)
    g = (target == class_id)

    intersection = (p & g).sum()
    denominator = p.sum() + g.sum()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
