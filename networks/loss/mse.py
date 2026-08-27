import numpy as np

from networks.loss.base import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self):
        self.cached_difference = None

    def forward(self, prediction, target):
        prediction = np.asarray(prediction)
        target = np.asarray(target)

        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        if prediction.size == 0:
            raise ValueError("prediction and target cannot be empty")

        self.cached_difference = prediction - target
        return np.mean(self.cached_difference ** 2)

    def backward(self):
        if self.cached_difference is None:
            raise RuntimeError("forward must be called before backward")

        return 2.0 * self.cached_difference / self.cached_difference.size
