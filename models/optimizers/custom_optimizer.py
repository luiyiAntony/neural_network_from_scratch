# models/optimizers/custom_optimizer.py
"""
# torch example
import torch.optim as optim

class CustomOptimizer:
    def __init__(self, parameters, lr=0.01):
        # Define custom optimizer initialization here
        self.optimizer = optim.SGD(parameters, lr=lr)  # Placeholder for a custom implementation
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
"""

class Optimizer:
    def __init__(self) -> None:
        # momentum
        self.momentum_v = None
        # Nesterov Accelerated Gradient (NAG)
        self.nesterov_v = None
        # AdaGrad
        self.AdaGrad_cache = None
        # RMSProp
        self.rms_cache = None
        # Adam ()
        self.adam_m = None
        self.adam_v = None

    def SGD(self, w, dw, lr=0.001):
        w += -lr * dw
        return w

    def momentum(self, w, dw, lr=0.001, mu=0.5): # commor mu in range [~0.5, 0.99]
        if self.momentum_v is None:
            self.momentum_v = np.zeros_like(w)
        self.momentum_v = mu * self.momentum_v - lr * dw
        w += self.momentum_v
        return w
 
    def NAG(self, w, dw, lr=0.001, mu=0.5): # Nesterov Accelerated Gradient
        if self.nesterov_v is None:
            self.nesterov_v = np.zeros_like(w)
        v_prev = self.nesterov_v
        self.nesterov_v = mu * self.nesterov_v - lr * dw
        w += -mu * v_prev + (1 + mu) * self.nesterov_v
        return w

    def AdaGrad(self, w, dw, lr=0.001): # Ada Grad
        if self.AdaGrad_cache is None:
            self.AdaGrad_cache = np.zeros_like(w)
        self.AdaGrad_cache += dw**2
        w += -lr * dw / (np.sqrt(self.AdaGrad_cache) + 1e-7)
        return w

    def RMSProp(self, w, dw, lr=0.001, decay_rate=0.99): # 
        if self.rms_cache is None:
            self.rms_cache = np.zeros_like(w)
        self.rms_cache = decay_rate * self.rms_cache + (1 - decay_rate) * dw**2
        w += -lr * dw / (np.sqrt(self.rms_cache) + 1e-7)
        return w
    
    def Adam(self, w, dw, lr=0.001, beta1=0.9, beta2=0.999, t=1): # 
        if self.adam_m is None:
            self.adam_m = np.zeros_like(w)
        if self.adam_v is None:
            self.adam_v = np.zeros_like(w)
        self.adam_m = beta1*self.adam_m + (1-beta1)*dw
        self.adam_v = beta2*self.adam_v + (1-beta2)*(dw**2)
        adam_m = self.adam_m / (1-beta1**t)
        adam_v = self.adam_v / (1-beta2**t)
        w += -lr*adam_m/(np.sqrt(adam_v)+1e-7)
        return w
