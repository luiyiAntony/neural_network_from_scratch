# tests/test_losses.py
import unittest
import torch

from models.losses.mse_loss import MSELoss
from models.losses.cross_entropy_loss import CrossEntropyLoss

class TestLosses(unittest.TestCase):
    def test_mse_loss(self):
        loss_fn = MSELoss()
        input = torch.tensor([0.0, 0.0, 0.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = loss_fn(input, target)
        expected_loss = ((input - target) ** 2).mean()
        self.assertAlmostEqual(loss.item(), expected_loss.item())
    
    def test_cross_entropy_loss(self):
        loss_fn = CrossEntropyLoss()
        input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.tensor([2, 1])
        loss = loss_fn(input, target)
        expected_loss = nn.CrossEntropyLoss()(input, target)
        self.assertAlmostEqual(loss.item(), expected_loss.item())

if __name__ == '__main__':
    unittest.main()
