from network.model import Model

def test_model_forward_backward():
    class DummyLayer:
        def forward(self, input):
            return input + 1

        def backward(self, upstream_gradient):
            return upstream_gradient * 2

    layer1 = DummyLayer()
    layer2 = DummyLayer()
    model = Model(layer1, layer2)

    assert len(model.layers) == 2

    input_data = 3
    forward_output = model.forward(input_data)
    assert forward_output == 5

    backward_input = 4
    backward_output = model.backward(backward_input)
    assert backward_output == 16


