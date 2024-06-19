# tests/test_layers.py
import sys
import os
import unittest
import numpy as np

from models.layers.dense import LinearLayer

class TestLayers(unittest.TestCase):
    def test_dense_layer(self):
        layer = LinearLayer(10, 5)
        x = np.random.randn(5, 10)
        output = layer.forward(x)
        self.assertEqual(output.shape, (5, 5))

if __name__ == '__main__':
    unittest.main()
