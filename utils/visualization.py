import sys
import numpy as np
from PIL import Image

from utils.data_processing import CIFAR10

def show_img(data, index):
    X = data.X
    y = data.y
    labels = data.label_names
    x = X[index]
    target = y[index]
    img = Image.fromarray(x, 'RGB')
    img.show()
    print(labels[target])


if __name__ == '__main__':
    args = sys.argv
    index_image = 0
    if '--index' in args:
        index_image = int(args[args.index('--index') + 1])
    data = CIFAR10(train=True)
    show_img(data, index_image)
