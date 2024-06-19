# tests/test_layers.py
import sys
import os
import unittest
import numpy as np

from models.layers.convolutional import ConvLayer
from models.layers.dense import LinearLayer
from models.layers.flatten import Flatten

"""
from models.layers.dense import DenseLayer
from models.layers.convolutional import ConvLayer
from models.layers.recurrent import LSTMLayer
from models.layers.activations import ReLUActivation, SigmoidActivation, TanhActivation
from models.layers.batchnorm import BatchNorm1d, BatchNorm2d
from models.layers.dropout import Dropout
"""

class TestLayers(unittest.TestCase):
    def test_dense_layer(self):
        layer = LinearLayer(10, 5)
        x = np.random.randn(5, 10)
        output = layer.forward(x)
        self.assertEqual(output.shape, (5, 5))
    
    def test_conv_layer(self):
        layer = ConvLayer(3, 6, 3)
        x = np.random.randn(10, 3, 32, 32) # 10 images RGB 32x32
        output = layer.forward(x)
        # forward test
        self.assertEqual(output.shape, (10, 6, 30, 30))
        # backward test
        grad_output = output # only for test porpuses
        grad_x = layer.backward(output)
        self.assertEqual(grad_x.shape, x.shape)

    def test_flatten(self):
        flatten = Flatten()
        input_test = np.array([
            [[1,2],
             [3,4]],
            [[5,6],
             [7,8]]
            ]) # (2, 2, 2)
        expected_output = np.array([
            [1,2,3,4],
            [5,6,7,8]
        ])
        # forward test
        output_test = flatten.forward(input_test)
        self.assertEqual(output_test.shape, expected_output.shape)
        # backward test
        grad_test = flatten.backward(expected_output)
        self.assertEqual(grad_test.shape, input_test.shape)
    
    """
    def test_lstm_layer(self):
        layer = LSTMLayer(10, 20)
        x = torch.randn(5, 10, 10)  # batch_size=5, seq_len=10, input_dim=10
        output, (hn, cn) = layer(x)
        self.assertEqual(output.shape, (5, 10, 20))

    def test_relu_activation(self):
        activation = ReLUActivation()
        x = torch.tensor([-1.0, 0.0, 1.0])
        output = activation(x)
        expected_output = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.equal(output, expected_output))
    
    def test_sigmoid_activation(self):
        activation = SigmoidActivation()
        x = torch.tensor([-1.0, 0.0, 1.0])
        output = activation(x)
        expected_output = torch.sigmoid(x)
        self.assertTrue(torch.allclose(output, expected_output))
    
    def test_tanh_activation(self):
        activation = TanhActivation()
        x = torch.tensor([-1.0, 0.0, 1.0])
        output = activation(x)
        expected_output = torch.tanh(x)
        self.assertTrue(torch.allclose(output, expected_output))
    
    def test_batchnorm1d(self):
        batchnorm = BatchNorm1d(3)
        x = torch.randn(10, 3)
        output = batchnorm(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_batchnorm2d(self):
        batchnorm = BatchNorm2d(3)
        x = torch.randn(10, 3, 32, 32)
        output = batchnorm(x)
        self.assertEqual(output.shape, x.shape)

    def test_dropout(self):
        dropout = Dropout(p=0.5)
        x = torch.randn(10, 3)
        output = dropout(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue((output == 0).sum().item() > 0)
    """

if __name__ == '__main__':
    unittest.main()
