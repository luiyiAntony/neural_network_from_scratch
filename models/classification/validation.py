import sys
import os
import pandas as pd
import numpy as np

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.layers.layer import Module
from models.layers.dense import LinearLayer
from models.layers.activations import ReLU
from models.losses.softmax_loss import SoftmaxLoss
from models.optimizers.sgd import SGDOptimizer
from utils.data_processing import CIFAR10, FashionMNIST, DataLoader

################################
# Create data loaders.
################################
# test data 
test_data = FashionMNIST(train=False)

################################
# Define model
################################
class ScratchNeuralNetwork(Module):
    def __init__(self):
        self.layers = [
            LinearLayer(28*28, 512),
            ReLU(),
            LinearLayer(512, 512),
            ReLU(),
            LinearLayer(512, 10)
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
# loading models
################################
model.load_model("pretrained/mnist.pkl")

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

i = 10
x, y = test_data.X[i], test_data.y[i]
pred = model.forward(x)
predicted, actual = classes[pred[0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
