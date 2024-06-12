import numpy as np

class MomentumOptimizer:
    def __init__(self, model, learning_rate=1e-3, mu=0.5):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum_v = {layer: {key: np.zeros_like(param) for key, param in layer.params.items()} for layer in model.layers if 'params' in layer.__dir__()}
        self.mu = mu

    def step(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                for key, param in layer.params.items():
                    self.momentum_v[layer][key] = self.mu * self.momentum_v[layer][key] - self.learning_rate * layer.grads[key]
                    param += self.momentum_v[layer][key]

    def zero_grad(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                layer.zero_grad()

