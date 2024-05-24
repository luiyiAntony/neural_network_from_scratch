# models/layers/activations.py
"""
# torch example
import torch.nn as nn

class ReLUActivation(nn.Module):
    def __init__(self):
        super(ReLUActivation, self).__init__()
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(x)

class SigmoidActivation(nn.Module):
    def __init__(self):
        super(SigmoidActivation, self).__init__()
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        return self.activation(x)

class TanhActivation(nn.Module):
    def __init__(self):
        super(TanhActivation, self).__init__()
        self.activation = nn.Tanh()
    
    def forward(self, x):
        return self.activation(x)
"""

# ACTIVATION FUNCTIONS
# Sigmoid
# tanh
# ReLU -> max(0, x)
# Leaky ReLU
# Maxout
# ELU


class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, x, test=False): # x : (n, m)
        self.x = np.array(x) # (n, m)
        self.activation = np.maximum(0, self.x) # (n, m)
        return self.activation 

    def backward(self, prev): # prev == dLoss/d_activation; prev.shape == d_activation.shape
        self.dx = np.zeros_like(self.x) # (n, m)
        self.dx[np.where(self.x > 0)] = 1 # (n, m)
        self.dx *= prev # (n, m)
        return self.dx

class LReLU:
    def __init__(self, alpha=0.001) -> None:
        self.alpha = alpha
        self.dalpha = None

    def forward(self, x): # x : (n, m)
        self.x = np.array(x) # (n, m)
        self.alpha_x = self.alpha * self.x # (n, m)
        self.activation = np.maximun(self.alpha_x, self.x) # (n, m)
        return self.activation 

    def backward(self, prev): # prev == dLoss/d_activation; prev.shape == d_activation.shape
        # when x > 0
        self.dx = np.zeros_like(self.x) # (n, m)
        self.dx[np.where(self.x > 0)] = 1 # (n, m)
        self.dx *= prev # (n, m)
        # when x < 0
        self.dalpha_x = np.zeros_like(self.x) # (n, m)
        self.dalpha_x[np.where(self.x < 0)] = 1 # (n, m)
        self.dalpha_x *= prev # (n, m)
        # 
        self.dx += np.ones_like(self.x) * self.alpha * self.dalpha_x # (n, m)
        self.dalpha = np.sum(self.x * self.dalpha_x) # INT
        return self.dx

    def update(self):
        self.alpha += self.dalpha
