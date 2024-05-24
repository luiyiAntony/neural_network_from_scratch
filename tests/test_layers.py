# tests/test_layers.py
import unittest
import torch

from models.layers.dense import DenseLayer
from models.layers.convolutional import ConvLayer
from models.layers.recurrent import LSTMLayer
from models.layers.activations import ReLUActivation, SigmoidActivation, TanhActivation
from models.layers.batchnorm import BatchNorm1d, BatchNorm2d

class TestLayers(unittest.TestCase):
    def test_dense_layer(self):
        layer = DenseLayer(10, 5)
        x = torch.randn(1, 10)
        output = layer(x)
        self.assertEqual(output.shape, (1, 5))
    
    def test_conv_layer(self):
        layer = ConvLayer(3, 6, 3)
        x = torch.randn(1, 3, 32, 32)
        output = layer(x)
        self.assertEqual(output.shape, (1, 6, 30, 30))
    
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

if __name__ == '__main__':
    unittest.main()
