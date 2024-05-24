# models/layers/batchnorm.py
"""
# torch example
import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm1d, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        return self.batchnorm(x)

class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        return self.batchnorm(x)
"""

class BatchNorm:
    def __init__(self, out_size, e=0.0001) -> None:
        self.alpha = np.ones((1, out_size))
        self.beta = np.zeros((1, out_size))
        self.e = e
        self.means = []
        self.vars = []

    def forward(self, X, test=False): # (n, m) -> n : len_batch, m : input_len
        self.X = X
        if test: # 
            self.mean = np.mean(np.array(self.means), axis=0)
            self.variance = np.mean(np.array(self.vars), axis=0)
        else: # 
            # mini-batch mean
            self.mean = np.sum(self.X, axis=0) / self.X.shape[0] # (m)
            self.means.append(self.mean)
            # mini-batch variance : 1/len_batch*(sum(xi - mean)**2)
            self.rest = self.X - self.mean # (n, m)
            self.rest_pow = self.rest ** 2 # (n, m)
            self.sum_rest_pow = np.sum(self.rest_pow, axis=0) # (m)
            self.variance = self.sum_rest_pow / self.X.shape[0] # (m)
            self.vars.append(self.variance)
        # normalize : (xi - mean)/(variance + e)**-2
        self.sqrt_variance = np.sqrt(self.variance + self.e) # (m)
        self.norm = (self.X - self.mean) /  self.sqrt_variance # (n, m)
        # scale and shift
        self.norm_batch = self.alpha * self.norm + self.beta # (1, m) * (n, m) + (1, m) = (n, m)
        return self.norm_batch

    def backward(self, prev): # (n, m)
        self.dalpha = np.sum(self.norm * prev, axis=0) # (m)
        self.dbeta = np.sum(prev, axis=0) # (m)
        self.dnorm = self.alpha * prev # (n, m)
        self.dX = (1/self.sqrt_variance) * self.dnorm # (n, m)
        self.dmean = np.sum((-1/self.sqrt_variance) * self.dnorm, axis=0) # (m)
        self.dsqrt_variance = np.sum((-(self.X - self.mean) / self.sqrt_variance ** 2) * self.dnorm, axis=0) # (m)
        self.dvariance = 0.5 * (self.variance ** -0.5) * self.dsqrt_variance # (m)
        self.dsum_rest_pow = self.X.shape[0] * self.dvariance # (m)
        self.drest_pow = np.ones_like(self.rest_pow) * self.dsum_rest_pow # (n, m)
        self.drest = 2*(self.rest) * self.drest_pow # (n, m)
        self.dX += np.ones_like(self.X) * self.drest # (n, m)
        self.dmean += -np.sum(self.drest, axis=0) # (m)
        self.dX += np.ones_like(self.X) * 1/self.X.shape[0] * self.dmean # (n, m)
        return self.dX

    def update(self, gradient_decent=0.001):
        self.alpha += -gradient_decent * self.dalpha
        self.beta  += -gradient_decent * self.dbeta

