# models/optimizers/adam.py
import torch.optim as optim

class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.optimizer = optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
