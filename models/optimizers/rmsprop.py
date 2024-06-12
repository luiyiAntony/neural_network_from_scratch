import numpy as np

class RMSprop:
    def __init__(self, model, learning_rate=1e-6, decay_rate=0.99, epsilon=1e-7):
        self.model = model
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.rms_cache = {layer: {key: np.zeros_like(param) for key, param in layer.params.items()} for layer in model.layers if 'params' in layer.__dir__()}
        self.epsilon = epsilon

    def step(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                for key, param in layer.params.items():
                    self.cache[layer][key]+= self.decay_rate * self.rms_cache + (1 - self.decay_rate) * layer.grads[key] ** 2
                    param += -self.learning_rate * layer.grads[key] / (np.sqrt(self.rms_cache[layer][key]) + self.epsilon)

    def zero_grad(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                layer.zero_grad()

