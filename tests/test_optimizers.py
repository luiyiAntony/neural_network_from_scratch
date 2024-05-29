# tests/test_optimizers.py
import unittest
import numpy as np
from models.layers.dense import LinearLayer
from models.optimizers.sgd import SGDOptimizer
from models.optimizers.adam import AdamOptimizer

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.model = LinearLayer(10, 5)
        self.parameters = self.model.x
    
    def test_sgd_optimizer(self):
        """
        optimizer = SGDOptimizer(self.parameters, lr=0.01)
        x = torch.randn(1, 10)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertIsNone(loss.grad)
        """
        pass
    
    def test_adam_optimizer(self):
        optimizer = AdamOptimizer(self.parameters, lr=0.001)
        x = torch.randn(1, 10)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertIsNone(loss.grad)

if __name__ == '__main__':
    unittest.main()
