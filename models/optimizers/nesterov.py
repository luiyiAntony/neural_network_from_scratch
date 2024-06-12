import numpy as np

class NesterovOptimizer:
    def __init__(self, model, learning_rate=1e-6, mu=0.5):
        self.model = model
        self.learning_rate = learning_rate

        self.nesterov_v = {layer: {key : np.zeros_like(param) for key, param in layer.params.items()} for layer in self.model.layers if 'params' in layer}
        self.mu = mu

    def step(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                for key, param in layer.params.items():
                    v_prev = self.nesterov_v[layer][key]
                    self.nesterov_v[layer][key] = self.mu * self.nesterov_v[layer][key] - self.learning_rate * layer.grads[key]
                    param += -self.mu * v_prev + (1 + self.mu) * self.nesterov_v[layer][key]

    def zero_grad(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                layer.zero_grad()
