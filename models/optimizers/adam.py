import numpy as np

# models/optimizers/adam.py
"""
import torch.optim as optim

class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.optimizer = optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
"""

class AdamOptimizer:
    def __init__(self, model, learning_rate=1e-6, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        """
        Conv1: W, b (width, height, num_kernels)
        layer1: W, b (input, output)
        relu: none
        layer2: W, b (input, output)
        relu: none
        SoftMax: none

        adam_m : [{"W", "b"}, ...]
        """

        self.adam_m = {layer: {key: np.zeros_like(param) for key, param in layer.params.items()} for layer in model.layers if 'params' in layer.__dir__()}
        self.adam_v = {layer: {key: np.zeros_like(param) for key, param in layer.params.items()} for layer in model.layers if 'params' in layer.__dir__()}
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.learning_rate * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))  

        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                for key, param in layer.params.items():
                    self.adam_m[layer][key] = self.beta1 * self.adam_m[layer][key] + (1 - self.beta1) * layer.grads[key]
                    self.adam_v[layer][key] = self.beta1 * self.adam_v[layer][key] + (1 - self.beta2) * (layer.grads[key] ** 2)
                    adam_m = self.adam_m[layer][key] / (1 - self.beta1 ** self.t)
                    adam_v = self.adam_v[layer][key] / (1 - self.beta2 ** self.t)
                    param += -lr_t * (adam_m / (np.sqrt(adam_v) - self.epsilon ))

    def zero_grad(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                layer.zero_grad()
