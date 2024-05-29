import sys
import os
import pandas as pd

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.layers.layer import Module
from models.layers.dense import LinearLayer
from models.layers.activations import ReLU
from models.losses.softmax_loss import SoftmaxLoss
from models.optimizers.sgd import SGDOptimizer
from utils.data_processing import CIFAR10, DataLoader

################################
# Create data loaders.
################################
# training data 
train_data = CIFAR10(train=True)

# test data 
test_data = CIFAR10(train=False)

################################
# Create data loaders.
################################
train_dataframe = pd.DataFrame(train_data.X)
train_dataframe['label'] = train_data.y
test_dataframe = pd.DataFrame(test_data.X)
test_dataframe['label'] = test_data.y
batch_size = 64
train_dataloader = DataLoader(train_dataframe,
                              batch_size=batch_size)
test_dataloader = DataLoader(test_dataframe,
                             batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, H * W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

################################
# Define model
################################
class ScratchNeuralNetwork(Module):
    def __init__(self):
        self.layers = [
            LinearLayer(32*32*3, 512),
            ReLU(),
            LinearLayer(512, 512),
            ReLU(),
            LinearLayer(512, 20)
        ]

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input # logits

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

model = ScratchNeuralNetwork()

################################
# Define the loss function
################################
loss_fn = SoftmaxLoss()

################################
# Define the optimizer
################################
optimizer = SGDOptimizer(model, lr=1e-3)

################################
# Define the train function
################################
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model.forward(X)
        loss = loss_fn.forward(pred, y)

        # Backpropagation
        logits_grad = loss_fn.backward()
        model.backward(logits_grad)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss, (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

################################
# Define the test function
################################
def test(dataloader, model, loss_fn):
    size = len(dataloader.data)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    for X, y in dataloader:
        pred = model(X)
        test_loss += loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

################################
# Training block
################################
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")






















