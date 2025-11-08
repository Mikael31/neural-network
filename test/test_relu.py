import numpy as np
from network.relu import ReLU

def test_forward_single():
    layer = ReLU()
    x = np.array([1.0, 2.0, 3.0, 0.0, -2.0])
    y = layer.forward(x)

    expected_output = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    assert np.array_equal(y, expected_output)

def test_forward_batch():
    layer = ReLU()
    x = np.array([[1.0, -1.0, 0.0], [-2.0, 3.0, 4.0]])
    y = layer.forward(x)

    expected_output = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    assert np.array_equal(y, expected_output)

def test_backward_single():
    layer = ReLU()
    x = np.array([1.0, -1.0, 0.0, 2.0])
    layer.forward(x)
    upstream_gradient = np.array([0.5, 0.5, 0.5, 0.5])
    dx = layer.backward(upstream_gradient)

    expected_gradient = np.array([0.5, 0.0, 0.0, 0.5])
    assert np.array_equal(dx, expected_gradient)

def test_backward_batch():
    layer = ReLU()
    x = np.array([[1.0, -1.0, 0.0], [-2.0, 3.0, 4.0]])
    layer.forward(x)
    upstream_gradient = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dx = layer.backward(upstream_gradient)

    expected_gradient = np.array([[0.1, 0.0, 0.0], [0.0, 0.5, 0.6]])
    assert np.array_equal(dx, expected_gradient)

