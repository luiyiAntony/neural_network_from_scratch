import numpy as np
import pickle

class Module:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, input):
        """
        Forward propagation
        Must be implemented in children classes
        """
        raise NotImplemented

    def backward(self, grad_output):
        """
        Backward propagation
        Must be implemented in children classes
        """
        raise NotImplemented

    def parameters(self):
        """
        Return a list of tuples (param, grad) to be used in the optimization
        """
        return [(self.params[key], self.grads[key]) for key in self.params]
    
    def zero_grad(self):
        """
        Set all the parameter's gradients to 0.
        """
        for key in self.params:
            self.grads[key] = np.zeros_like(self.grads[key])

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self = pickle.load(f)
