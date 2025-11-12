import numpy as np

class LinearLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.init_weights()

    def init_weights(self, activation="relu"):
        if activation == "relu":
            std_dev = np.sqrt(2 / self.input_size)
        else:
            std_dev = np.sqrt(2 / self.input_size + self.output_size)

        self.weights = np.random.normal(0, std_dev, (self.output_size, self.input_size))
        self.biases = np.zeros(self.output_size)

    def forward(self, input):
        self.cached_input = input
        return input @ self.weights.T + self.biases

    def backward(self, upstream_gradient):
        upstream_gradient = np.atleast_2d(upstream_gradient)
        self.cached_input = np.atleast_2d(self.cached_input)

        self.dw = upstream_gradient.T @ self.cached_input
        self.db = upstream_gradient.T @ np.ones(upstream_gradient.shape[0])
        dx = upstream_gradient @ self.weights
        return dx if upstream_gradient.shape[0] > 1 else dx.squeeze()

    def parameters(self):
        return [self.weights, self.biases]

    def gradients(self):
        return [self.dw, self.db]

