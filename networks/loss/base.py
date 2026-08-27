from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def forward(self, prediction, target):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

    def __call__(self, prediction, target):
        return self.forward(prediction, target)
