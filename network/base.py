from abc import ABC, abstractmethod
import numpy as np

class BaseLayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, input) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, upstream_gradient) -> np.ndarray:
        pass

