"""

"""

import numpy as np
from model.modified_layers import Conv2D, DenseLayer


class NeuralNetwork:
    def __init__(self, layers):
        self.LAYERS = layers

    def forward_propagation(self, inputs: np.array):
        for layer in self.LAYERS:
            inputs = layer.forward_propagation(inputs)
        return inputs

    def load(self, filepath: str):
        params = np.load(filepath)
        layer_index = 0

        for layer in self.LAYERS:
            if isinstance(layer, Conv2D) or isinstance(layer, DenseLayer):
                layer.weights = params[f"layer_{layer_index}_weights"]
                layer.biases = params[f"layer_{layer_index}_biases"]
            layer_index += 1


