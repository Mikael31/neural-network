import numpy as np

class AdamOptimizer():
    def __init__(self, model, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [[np.zeros_like(param) for param in layer.parameters()] for layer in self.model.layers]
        self.v = [[np.zeros_like(param) for param in layer.parameters()] for layer in self.model.layers]
        self.t = 1

    def step(self):
        for i, layer in enumerate(self.model.layers):
            for j, (param, grad) in enumerate(zip(layer.parameters(), layer.gradients())):
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)

                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.t += 1

