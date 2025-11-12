class Model:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, input):
        upstream_gradient = input
        for layer in reversed(self.layers):
            upstream_gradient = layer.backward(upstream_gradient)
        return upstream_gradient

