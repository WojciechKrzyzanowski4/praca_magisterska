import numpy as np

from networks.optimizer.base import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, parameters, learning_rate, momentum=0.9):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in the range [0, 1)")

        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        for parameter, velocity in zip(self.parameters, self.velocity):
            velocity[...] = self.momentum * velocity + parameter.grad
            parameter.data[...] -= self.learning_rate * velocity

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.fill(0)
