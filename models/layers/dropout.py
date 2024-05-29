# models/layers/dropout.py

"""
# torch example
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        return self.dropout(x)
"""

class Dropout:
    def __init__(self) -> None:
        pass

    # x : input (batch)
    # p : percent of dropout
    def forward(self, x, p=0.5, test=False):
        if test:
            x *= p # (n, m)
            return x
        else:
            self.U = np.random.rand(*x.shape) < p # (n, m)
            x *= self.U # (n, m)
            return x

    def backward(self, prev): # x.shape == prev.shape
        self.dx = self.U * prev # (n, m)
        return self.dx
