
import sys
import numpy as np
from PIL import Image

from data_processing import CIFAR10

def show_img(data, index):
    X = data.X
    y = data.y
    labels = data.label_names
    x = X[index]
    target = y[index]
    # x : (3073) (need to convert to) -> (32 x 32 x 3)
    # first chunk  : 1024 red channel values
    # second chunk : 1024 green channel values
    # third chunk  : 1024 blue channel values
    red = x[0:1024] # red values
    green = x[1024:2048] # green values
    blue = x[2048:3072] # blue values
    # mix (r,g,b) values
    rs_img = np.array([[red[i],green[i],blue[i]] for i in range(1024)])
    rs_img = np.reshape(rs_img, ( 32, 32, 3))
    img = Image.fromarray(rs_img, 'RGB')
    img.show()
    print(labels[target])
    #print(np.array(labels)[labels[index]])


if __name__ == '__main__':
    args = sys.argv
    index_image = 0
    if '--index' in args:
        index_image = int(args[args.index('--index') + 1])
    data = CIFAR10(train=True)
    show_img(data, index_image)
