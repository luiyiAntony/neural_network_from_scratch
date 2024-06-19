import sys
import os
import pandas as pd
import numpy as np

from models.layers.layer import Module
from models.layers.dense import LinearLayer
from models.layers.flatten import Flatten
from models.layers.convolutional import ConvLayer
from models.layers.activations import ReLU
from models.losses.softmax_loss import SoftmaxLoss
from models.optimizers.sgd import SGDOptimizer
from models.optimizers.adam import AdamOptimizer
from models.optimizers.adagrad import AdagradOptimizer
from utils.data_processing import CIFAR10, FashionMNIST, DataLoader

################################
# Get data sets
################################
# training data 
train_data = CIFAR10(train=True)

# test data 
test_data = CIFAR10(train=False)

################################
# Create data loaders.
################################
# train_dataframe = pd.DataFrame(train_data.X)
# train_dataframe['label'] = train_data.y
# test_dataframe = pd.DataFrame(test_data.X)
# test_dataframe['label'] = test_data.y
batch_size = 64
train_dataloader = DataLoader(train_data,
                              batch_size=batch_size)
test_dataloader = DataLoader(test_data,
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
            ConvLayer(in_channels=3, out_channels=5, kernel_size=3),
            ConvLayer(in_channels=5, out_channels=10, kernel_size=3),
            Flatten(),
            LinearLayer(28*28*10, 512),
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
#optimizer = SGDOptimizer(model, lerning_rate=1e-3)
optimizer = AdagradOptimizer(model, learning_rate=1e-8)

################################
# Define the train function
################################
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.data.X)
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
    size = len(dataloader.data.X)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    for X, y in dataloader:
        pred = model.forward(X)
        test_loss += loss_fn.forward(pred, y)
        correct += (pred.argmax(1) == y).astype(np.float64).sum().item()
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

################################
# Saving Models
################################
model.save_model("pretrained/simple_network.pkl")
print("Saved PyTorch Model State to model.pth")

################################
# loading models
################################
model.load_model("pretrained/simple_network.pkl")

################################
# Prediction
################################
classes = [
    'aquatic_mammals', 
    'fish', 
    'flowers', 
    'food_containers', 
    'fruit_and_vegetables', 
    'household_electrical_devices', 
    'household_furniture', 
    'insects', 
    'large_carnivores', 
    'large_man-made_outdoor_things', 
    'large_natural_outdoor_scenes', 
    'large_omnivores_and_herbivores', 
    'medium_mammals', 
    'non-insect_invertebrates', 
    'people', 
    'reptiles', 
    'small_mammals', 
    'trees', 
    'vehicles_1', 
    'vehicles_2'
]

x, y = test_data.X[0], test_data.y[0]
pred = model.forward(x)
predicted, actual = classes[pred[0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')




















