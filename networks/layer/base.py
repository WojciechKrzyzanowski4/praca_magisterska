from abc import ABC, abstractmethod


class BaseLayer(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)