# models/optimizers/sgd.py
"""
# torch example
import torch.optim as optim

class SGDOptimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
"""

class SGDOptimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if 'params' in layer.__dir__():
                for param, grad in layer.parameters():
                    param += -self.lr * grad

    def zero_grad(self):
        for layer in self.model.layers:
            layer.zero_grad()
