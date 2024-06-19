import numpy as np
from models.layers.layer import Module

class Flatten(Module):
    def __init__(self):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        batch_size = self.input_shape[0]
        flatten_input = np.array(input)
        flatten_input = np.reshape(flatten_input, (batch_size,-1))
        return flatten_input
        
    def backward(self, grad_output):
        grad = np.array(grad_output)
        input_grad = np.reshape(grad, self.input_shape)
        return input_grad

