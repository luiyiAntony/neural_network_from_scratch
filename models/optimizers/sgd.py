# models/optimizers/sgd.py
import torch.optim as optim

class SGDOptimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
