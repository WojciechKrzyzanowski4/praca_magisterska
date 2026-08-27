import numpy as np

from networks.loss.base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        self.cached_probabilities = None
        self.cached_targets = None

    def forward(self, logits, targets):
        logits = np.asarray(logits)
        targets = np.asarray(targets)

        if logits.ndim != 2:
            raise ValueError("logits must have shape (batch, classes)")
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("logits cannot be empty")
        if targets.shape != (logits.shape[0],):
            raise ValueError("targets must have shape (batch,)")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("targets must contain integer class indices")
        if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
            raise ValueError("target class index is outside the logits range")

        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
        probabilities = exp_logits / sum_exp_logits
        log_probabilities = shifted_logits - np.log(sum_exp_logits)

        batch_indices = np.arange(logits.shape[0])
        correct_class_log_probabilities = log_probabilities[
            batch_indices,
            targets,
        ]

        self.cached_probabilities = probabilities
        self.cached_targets = targets.copy()

        return -np.mean(correct_class_log_probabilities)

    def backward(self):
        if self.cached_probabilities is None or self.cached_targets is None:
            raise RuntimeError("forward must be called before backward")

        gradient = self.cached_probabilities.copy()
        batch_indices = np.arange(gradient.shape[0])
        gradient[batch_indices, self.cached_targets] -= 1.0
        gradient /= gradient.shape[0]
        return gradient
