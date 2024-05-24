# models/losses/cross_entropy_loss.py
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input, target):
        return self.loss(input, target)
