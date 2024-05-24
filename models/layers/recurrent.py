# models/layers/recurrent.py
import torch
import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)
