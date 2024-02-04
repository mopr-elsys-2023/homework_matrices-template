import functools
import numpy as np

from src.activation_functions import sigmoid, softmax, relu, tanh, flatten

activation_functions = {
    'sigmoid': sigmoid,
    'softmax': softmax,
    'relu': relu,
    'tanh': tanh,
    'flatten': flatten
}


def forward_propagation(layers):

    def apply_layer(matrix, layer):
        
        activation_function_name = layer['activation']
        activation_function = activation_functions[activation_function_name]

        next_layer_matrix = None

        if 'weights' not in layer and 'bias' not in layer:
            next_layer_matrix = activation_function(matrix)
        elif 'weights' in layer and 'bias' in layer:
            
            weights = layer['weights']
            bias = layer['bias']

            next_layer_matrix = np.matmul(matrix, weights)

            next_layer_matrix += bias

            return activation_function(next_layer_matrix)


        return next_layer_matrix
        
    return lambda matrix: functools.reduce(apply_layer, layers, matrix)
