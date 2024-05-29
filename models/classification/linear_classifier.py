import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import uniform, lognormvariate

from models.layers.dense import LinearLayer

#import torch


np.random.seed(42)

class NeuralNetwork: # arquitectura de la NN
    def __init__(self, input_size, n_hidden, out_size) -> None:
        # CREATE LAYERS
        self.linearLayer1 = LinearLayer(input_size, n_hidden, xavier_init=False, 
                                        init_factor=0.0001, name="linear layer 1", ensembles=True) # Linear layer
        self.relu = ReLU() # non linearity layer
        self.dropout = Dropout() # regularization
        self.linearLayer2 = LinearLayer(n_hidden, out_size, xavier_init=False, 
                                        init_factor=0.0001, name="linear layer 2", ensembles=True) # Linear layer
        self.softmax = SoftmaxLoss() # softmax layer
        self.layers = [self.linearLayer1, self.relu, self.dropout, self.linearLayer2]
    
    def train(self, X_train, Y_train, X_test, Y_test, 
              steps=100, lr=0.001, batch_size=100, learning_rate_decay=0.1):

        # TRAKING
        _loss = []
        _accuracy = []
        ratweight_updates_magnitudes = {}
        for layer in self.layers:
            if "w" in dir(layer):
                ratweight_updates_magnitudes[layer.name] = []

        for i in range(steps):
            # 1/t decay: lr = lr/(1+decay*t)
            # exponention decay : lr = lr * exp(-decay*t)
            # update learning-rate each 50 steps (step decay)
            if steps % 1 == 0:
                lr *= learning_rate_decay
            # create train batch
            rand_indexs = np.random.randint(0, len(X_train), size=batch_size)
            X = X_train[rand_indexs]
            Y = np.array(Y_train)[rand_indexs]

            # FORWARD
            output = X
            for layer in self.layers:
                output = layer.forward(output)
            loss = self.softmax.forward(output, Y) # SOFTMAX
            _loss.append(loss)

            # BACKWARD
            doutput = self.softmax.backward() # SOFTMAX
            for layer in reversed(self.layers):
                doutput = layer.backward(doutput)

            # UPDATE
            for layer in reversed(self.layers):
                if "update" in dir(layer):
                    layer.update(lr=lr)
            
            # traking
            for layer in self.layers:
                if "w" in dir(layer):
                    param_scale = np.linalg.norm(layer.w)
                    update_scale = np.linalg.norm(layer.dw)
                    ratio = update_scale / param_scale # want ~1e-3
                    #print(f"ratio layer {layer.name} param_scale / update_scale : {ratio}")
                    ratweight_updates_magnitudes[layer.name].append(ratio)

            # ACCURACY VALIDATION
            if i % 100 == 0:
                X_val = X_test
                Y_val = Y_test
                output = X_val
                for layer in self.layers:
                    output = layer.forward(output, test=True)
                probs = self.softmax.forward(output, Y_val, accuracy=True) # (n, m)
                predicts = np.argmax(probs, axis=1) # (n)
                n_well = np.sum(np.equal(predicts, Y_val))
                val_acc = n_well/len(predicts)
                _accuracy.append(val_acc)

        # PLOT
        draw(_loss, file_name="loss", xlabel="steps", ylabel="loss")
        draw(_accuracy, file_name="accuracy", xlabel="steps", ylabel="accuracy")
        for layer in self.layers:
            if "w" in dir(layer):
                data = ratweight_updates_magnitudes[layer.name]
                draw(data, file_name="ratio_up_we"+layer.name, xlabel="steps", ylabel="ratio_up_we")
        return loss, val_acc

    
def draw(data, file_name, dir="imgs", xlabel="xlabel", ylabel="ylabel"):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(dir + "/" + file_name)
    plt.close()

if __name__ == '__main__':
    np.random.seed(1)
    # create test data
    #x = np.ones((32, 32, 3), dtype=np.float32)
    #x_torch = torch.ones((3, 32, 32), requires_grad=True, dtype=torch.float32)
    x = np.random.randn(6, 6, 3)
    x_torch = torch.ones((3, 6, 6), requires_grad=True, dtype=torch.float32)
    x_torch.data = torch.tensor([x[:,:,i] for i in range(3)], dtype=torch.float32)
    #print(x)
    #print(x_torch)
    #exit(1)
    # conv
    conv = Conv(3, 3, 5)
    conv_torch = torch.nn.Conv2d(3, 5, kernel_size=3, dtype=torch.float32, bias=False)
    conv_torch.weight.data = torch.tensor(conv.kernels, dtype=torch.float32)
    #print(conv.kernels)
    #print(conv.kernels.dtype)
    #print(conv_torch.weight.data[1])
    #print(conv_torch.weight.dtype)
    #print(conv.kernels.data == conv_torch.weight.data)
    #exit(1)
    # forward
    result = conv.forward(x)
    result_torch = conv_torch.forward(x_torch)
    loss = torch.sum(result_torch)
    loss.backward()
    #print(result)
    #print(result.shape)
    #print(result_torch)
    #print(result_torch.shape)
    #exit(1)
    back = conv.backward(np.ones_like(result))
    #print(f" conv.d_k: {conv.d_k[1,:,:,:]}")
    #print(f" conv.d_k.shape: {conv.d_k.shape}")
    #print(f" conv.d_x: {conv.d_x[:,:,1]}")
    print(f" conv.d_x: {conv.d_x}")
    print(f" conv.d_x.shape: {conv.d_x.shape}")

    #print(f" conv_torch.weight.grad: {conv_torch.weight.grad[1]}")
    #print(f" conv_torch.weight.shape: {conv_torch.weight.grad.shape}")
    #print(f" x_torch: {x_torch.grad[1]}")
    print(f" x_torch: {x_torch.grad}")
    print(f" x_torch.shape: {x_torch.grad.shape}")
    exit(1)
    
    # extract data
    coarse_labels_train, train, coarse_labels_test, test, coarse_label_names = load_data()
    #show_img(train, index=102)

    # PREPROCESS DATA
    train = np.array(train, dtype=float)
    test = np.array(test, dtype=float)
    # center data (mean = ~0)
    train -= np.mean(train, axis=0)
    test  -= np.mean(test , axis=0)
    # normalized data (std = ~1)
    train /= np.std(train, axis=0)
    test  /= np.std(test , axis=0)
    # extract the dimensions
    input_size = len(train[0])
    out_size = len(coarse_label_names)

    # HYPER-PARAMS
    #learning_rate = 10**lognormvariate(-3,-6)
    learning_rate = 0.001
    n_hidden = 50
    steps = 2000
    learning_rate_decay = 1

    # TRAINING
    model = Model(input_size=input_size, n_hidden=n_hidden, out_size=out_size)
    loss, val_acc = model.train(
        X_train=train, Y_train=coarse_labels_train, 
        X_test=test, Y_test=coarse_labels_test, 
        steps=steps, lr=learning_rate, batch_size=32,
        learning_rate_decay=learning_rate_decay)
    print(f"loss: {'{0:.4}'.format(loss)}, val_acc: {'{0:.4}'.format(val_acc)}")
    """
    # HYPER-PARAMETERS OPTIMIZER
    max_count = 100
    for i in range(max_count):
        # HYPER-PARAMS
        #learning_rate = 10**lognormvariate(-3,-6)
        learning_rate = 10**uniform(-3,-6)
        n_hidden = 50
        steps = 5 

        # TRAINING
        model = Model(input_size=input_size, n_hidden=n_hidden, out_size=out_size)
        val_acc = model.train(
            X_train=train, Y_train=coarse_labels_train, 
            X_test=train, Y_test=coarse_labels_test, 
            steps=steps, lr=learning_rate, batch_size=100)
        print(f"val_acc: {'{0:.6}'.format(val_acc)}, lr: {'{:.6f}'.format(learning_rate)}, ({i+1}/{max_count})")
    """
