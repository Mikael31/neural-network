from .base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.cached_input = input
        return np.maximum(0, input)

    def backward(self, upstream_gradient):
        mask = self.cached_input > 0
        masked_gradient = upstream_gradient * mask
        return masked_gradient

