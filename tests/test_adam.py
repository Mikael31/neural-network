from network.adam import AdamOptimizer
import numpy as np

def test_initialization():
    class DummyModel:
        def __init__(self):
            self.layers = [self.DummyLayer(), self.DummyLayer()]

        class DummyLayer:
            def parameters(self):
                return [0.5, -0.5]

            def gradients(self):
                return [0.1, -0.1]

    model = DummyModel()
    optimizer = AdamOptimizer(model, learning_rate=0.001)

    assert optimizer.learning_rate == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.epsilon == 1e-8
    assert len(optimizer.m) == 2
    assert len(optimizer.v) == 2
    assert optimizer.t == 1

def test_momentum_shape():
    class DummyModel:
        def __init__(self):
            self.layers = [self.DummyLayer(), self.DummyActivationLayer(), self.DummyLayer()]

        class DummyLayer:
            def parameters(self):
                return [np.array([0.5, -0.5]), np.array([1.0, -1.0])]

            def gradients(self):
                return [np.array([0.1, -0.1]), np.array([0.2, -0.2])]

        class DummyActivationLayer:
            def parameters(self):
                return []

            def gradients(self):
                return []

    model = DummyModel()
    optimizer = AdamOptimizer(model, learning_rate=0.001)
    assert len(optimizer.m) == 3
    assert len(optimizer.m[0]) == 2
    assert len(optimizer.m[1]) == 0
    assert len(optimizer.m[2]) == 2
    assert optimizer.m[0][0].shape == (2,)
    assert optimizer.m[0][1].shape == (2,)
    assert optimizer.m[2][0].shape == (2,)
    assert optimizer.m[2][1].shape == (2,)

    assert len(optimizer.v) == 3
    assert len(optimizer.v[0]) == 2
    assert len(optimizer.v[1]) == 0
    assert len(optimizer.v[2]) == 2
    assert optimizer.v[0][0].shape == (2,)
    assert optimizer.v[0][1].shape == (2,)
    assert optimizer.v[2][0].shape == (2,)
    assert optimizer.v[2][1].shape == (2,)

def test_step():
    class DummyModel:
        def __init__(self):
            self.layers = [self.DummyLayer(), self.DummyLayer()]

        class DummyLayer:
            def __init__(self):
                self.params = [np.array([0.5, -0.5]), np.array([1.0, -1.0])]
                self.grads = [np.array([0.1, -0.1]), np.array([0.2, -0.2])]

            def parameters(self):
                return self.params

            def gradients(self):
                return self.grads

    model = DummyModel()
    optimizer = AdamOptimizer(model, learning_rate=0.0001)
    optimizer.step()

    assert optimizer.t == 2
    assert not np.array_equal(model.layers[0].params[0], np.array([0.5, -0.5]))
    assert not np.array_equal(model.layers[0].params[1], np.array([1.0, -1.0]))
    assert not np.array_equal(model.layers[1].params[1], np.array([1.0, -1.0]))
    assert not np.array_equal(model.layers[1].params[0], np.array([0.5, -0.5]))

