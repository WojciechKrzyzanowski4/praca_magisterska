from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError
