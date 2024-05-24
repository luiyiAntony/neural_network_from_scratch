import numpy as np

from models.optimizers.custom_optimizer import Optimizer

# models/layers/dense.py
"""
#torch example
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
"""

class LinearLayer:
    """
    input_size : number of parameters of each neuron
    output_size : number of neurons
    """
    def __init__(self, input_size, output_size, xavier_init=False, init_factor = 1, name="linear layer", ensembles=False):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.x = None # needed for the backprop
        self.w = np.random.randn(self.input_size, self.output_size) * init_factor
        # Xavier initialization
        if xavier_init:
            self.w = self.w / np.sqrt(input_size / 2) # 
        self.b = np.zeros(self.output_size) # initialize with 0.001 if the next layer is ReLU
        self.xw = None # needed for the backprop
        self.dx = None # 
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.optimizer1 = Optimizer()
        self.optimizer2 = Optimizer()
        self.step = 0
        # Emsembles
        self.ensembles = ensembles
        if self.ensembles:
            self.w_test = self.w
            self.b_test = self.b

    def forward(self, x, test=False):
        """
        x : input matrix
        """
        if test and self.ensembles:
            self.x = np.array(x) # needed for the back propagation; x = ( any, input_size)
            self.xw = x @ self.w_test + self.b_test  # (any, input_size) @ (input_size, output_size) = (any, output_size)
            return self.xw
        else:
            self.x = np.array(x) # needed for the back propagation; x = ( any, input_size)
            self.xw = x @ self.w + self.b  # (any, input_size) @ (input_size, output_size) = (any, output_size)
            return self.xw

    def backward(self, prev): # prev == dLoss/dxw; prev.shape == xw.shape
        self.dx = prev @ self.w.transpose()  # (any, output_size) @ (output_size, input_size) = (any, input_size)
        self.dw = self.x.transpose() @ prev # (input_size, any) @ (any, output_size) = (input_size, ouput_size)
        self.db = np.sum(prev) # 
        return self.dx
    
    def update(self, lr, optimizer="SGD"):
        # SGD (lr: 0.1 , loss: 2.477, val_acc: 0.2254) (lr: 0.001 , loss: 2.996, val_acc: 0.1253)
        #self.w = self.optimizer1.SGD(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.SGD(self.b, self.db, lr=lr)
        # momentum (lr: 0.1 , loss: 2.567, val_acc: 0.1964) (lr: 0.001 , loss: 2.996, val_acc: 0.1059)
        #self.w = self.optimizer1.momentum(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.momentum(self.b, self.db, lr=lr)
        # ANG (lr: 0.1 , loss: 2.8483 , val_acc: 0.1876)
        #self.w = self.optimizer1.NAG(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.NAG(self.b, self.db, lr=lr)
        # AdaGrad (lr: 0.1, loss: NaN , val_acc: 0.5) (lr: 0.001, loss: 2.523 , val_acc: 0.1996)
        #self.w = self.optimizer1.AdaGrad(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.AdaGrad(self.b, self.db, lr=lr)
        # RMSProp (lr: 0.1 , loss: nan, val_acc: 0.05) (lr: 0.001 , loss: 2.405 , val_acc: 0.2658)
        # (lr: 0.001, ensembles=True, loss: 2.405 , val_acc: 0.2912)
        #self.w = self.optimizer1.RMSProp(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.RMSProp(self.b, self.db, lr=lr)
        # IF USE ADAM (don't neet learning grade decay)
        # (lr: 0.1 , loss: nan, val_acc: 0.05) (lr: 0.001, loss: 2.989, val_acc: 0.0587)
        # (lr: 0.001, ensembles=True, loss: 2.305 , val_acc: 0.2511)
        self.step += 1
        self.w = self.optimizer1.Adam(self.w, self.dw, lr=lr, t=self.step)
        self.b = self.optimizer2.Adam(self.b, self.db, lr=lr, t=self.step)
        # ENSEMBLES
        if self.ensembles:
            self.w_test = 0.995*self.w_test + 0.005*self.w
            self.b_test = 0.995*self.b_test + 0.005*self.b

class LinearTest:
    def __init__(self, input, output):
        self.w = np.random.randn(input, output)

    def forward(self, x):
        """
        x : (anything, input)
        """
        return x @ self.w # (anything, output)

