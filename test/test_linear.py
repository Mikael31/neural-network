import numpy as np
from network.linear import LinearLayer

def test_init_weights_shape():
    layer = LinearLayer(4, 3)
    assert layer.weights.shape == (3, 4)
    assert layer.biases.shape == (3,)

def test_forward_shape_single():
    input_size, output_size = 4, 2
    layer = LinearLayer(input_size, output_size)
    x = np.random.randn(input_size)

    y = layer.forward(x)
    assert y.shape == (output_size,)

def test_forward_shape_batch():
    batch_size, input_size, output_size = 5, 4, 3
    layer = LinearLayer(input_size, output_size)
    x = np.random.randn(batch_size, input_size)

    y = layer.forward(x)

    assert y.shape == (batch_size, output_size)

def test_backward_shapes_single():
    in_size, out_size = 4, 2
    layer = LinearLayer(in_size, out_size)
    x = np.random.randn(in_size)
    upstream_gradient = np.random.randn(out_size)

    layer.forward(x)
    dx = layer.backward(upstream_gradient)

    assert dx.shape == (in_size,)
    assert layer.dw.shape == (out_size, in_size)
    assert layer.db.shape == (out_size,)


def test_backward_shapes_batch():
    batch_size, in_size, out_size = 5, 4, 3
    layer = LinearLayer(in_size, out_size)
    x = np.random.randn(batch_size, in_size)
    upstream_gradient = np.random.randn(batch_size, out_size)

    layer.forward(x)
    dx = layer.backward(upstream_gradient)

    assert dx.shape == (batch_size, in_size)
    assert layer.dw.shape == (out_size, in_size)
    assert layer.db.shape == (out_size,)

