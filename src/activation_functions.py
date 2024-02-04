import numpy as np

def sigmoid(matrix):
    def sigmoid_element(x):
        return np.exp(x) / (1 + np.exp(x))

    return np.vectorize(sigmoid_element)(matrix)

def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=0)

def tanh(matrix):
    def tanh_element(x):
        return ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )

    return np.vectorize(tanh_element)(matrix)

def relu(matrix):
    def relu_element(x):
        return max(x, 0)

    return np.vectorize(relu_element)(matrix)

def flatten(matrix):
    return matrix.flatten()
