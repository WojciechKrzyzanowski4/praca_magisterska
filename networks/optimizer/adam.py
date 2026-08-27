import numpy as np

from networks.optimizer.base import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(
        self,
        parameters,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in the range [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in the range [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_number = 0
        self.first_moment = [
            np.zeros_like(parameter.data) for parameter in self.parameters
        ]
        self.second_moment = [
            np.zeros_like(parameter.data) for parameter in self.parameters
        ]

    def step(self):
        self.step_number += 1

        for parameter, first_moment, second_moment in zip(
            self.parameters,
            self.first_moment,
            self.second_moment,
        ):
            first_moment[...] = (
                self.beta1 * first_moment + (1 - self.beta1) * parameter.grad
            )
            second_moment[...] = (
                self.beta2 * second_moment
                + (1 - self.beta2) * parameter.grad ** 2
            )

            corrected_first = first_moment / (1 - self.beta1 ** self.step_number)
            corrected_second = second_moment / (1 - self.beta2 ** self.step_number)
            parameter.data[...] -= (
                self.learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.fill(0)
