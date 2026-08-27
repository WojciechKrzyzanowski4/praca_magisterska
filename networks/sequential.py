from networks.layer.base import BaseLayer


class Sequential:

    def __init__(self, *layers):
        self.layers: list[BaseLayer] = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters(self):
        result = []
        for layer in self.layers:
            result.extend(layer.parameters())
        return result