# tests/tests_dataset.py
import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processing import CIFAR10, FashionMNIST, DataLoader

class TestDataset(unittest.TestCase):
    def test_train_split_dataset(self):
        dataset = CIFAR10(
            root = "", 
            train = True
        )
        self.assertEqual(dataset.X.shape, (50000, 32, 32, 3), msg="INCORRECT SHAPE (X.shape)")
        self.assertEqual(len(dataset.y), 50000, msg="INCORRECT NUMBER OF LABELS (len(y))")
        self.assertEqual(len(dataset.label_names), 20, msg="INCORRECT NUMBER OF NAME LABELS")

    def test_test_split_dataset(self):
        dataset = CIFAR10(
            root = "", 
            train = False
        )
        self.assertEqual(dataset.X.shape, (10000, 32, 32, 3), msg="INCORRECT SHAPE (X.shape)")
        self.assertEqual(len(dataset.y), 10000, msg="INCORRECT NUMBER OF LABELS (len(y))")
        self.assertEqual(len(dataset.label_names), 20, msg="INCORRECT NUMBER OF NAME LABELS")

    def test_mnist_dataset(self):
        # TRAIN
        dataset = FashionMNIST(
            root = "", 
            train = True
        )
        self.assertEqual(dataset.X.shape, (60000, 28, 28, 1), msg="INCORRECT SHAPE (X.shape)")
        self.assertEqual(len(dataset.y), 60000, msg="INCORRECT NUMBER OF LABELS (len(y))")
       
        # TEST
        dataset = FashionMNIST(
            root = "", 
            train = False
        )
        self.assertEqual(dataset.X.shape, (10000, 28, 28, 1), msg="INCORRECT SHAPE (X.shape)")
        self.assertEqual(len(dataset.y), 10000, msg="INCORRECT NUMBER OF LABELS (len(y))")

    def test_dataloader(self):
        dataset = FashionMNIST(root="", train=True)
        batchsize = 16
                            
        # Create a DataLoader with batch size 16
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        self.assertEqual(len(dataloader), dataset.X.shape[0]/batchsize)
       
        # Iterate through batches
        for batch, (X, y) in enumerate(dataloader):
            inputs = np.array(X)
            targets = np.array(y)
            self.assertEqual(inputs.shape, (batchsize, *(dataset.X.shape[1:]))) # avoid the first axis of the X's shape
            self.assertEqual(targets.shape, (batchsize,))
            if batch > 2:
                break

        # Access a specific batch by index
        inputs, tergets = dataloader[1]
        self.assertEqual(inputs.shape, (batchsize, *(dataset.X.shape[1:]))) # avoid the first axis of the X's shape
        self.assertEqual(targets.shape, (batchsize,))

        

if __name__ == '__main__':
    unittest.main()

