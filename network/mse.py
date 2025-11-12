import numpy as np

class MSELoss:
    def forward(self, predicted, truth):
        self.predicted = predicted
        self.truth = truth
        # Fully reduced loss, meaning average of batch aswell
        return np.mean((predicted - truth) ** 2)

    def backward(self):
        # 1/n * ((y_0 - y^hat_0)^2 + ... + (y_n - y^hat_n)^2)
        # dL/dy^hat = 1/n * (2 * (y^hat_0 - y_0) * 1 + ... + 2 * (y^hat_n - y_n) * 1)
        # = 2/n * (y^hat_0 - y_0 + ... + y^hat_n - y_n)

        return (2 / self.predicted.size) * (self.predicted - self.truth)

