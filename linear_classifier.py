import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import uniform, lognormvariate

import torch

np.random.seed(42)

def load_data():
    import pickle
    data_dir = 'data/'
    def load_chunk(file_name):
        with open(data_dir + file_name, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        # meta: fine_label_names, coarse_label_names
        # train = test: filenames, batch_label, fine_labels, coarse_labels, data
        #for i, d in enumerate(dict):
            #print(d)
            #print(d, dict[d])
        return dict
    # label's name
    meta = load_chunk('meta')
    fine_label_names = meta[b'fine_label_names']
    coarse_label_names = meta[b'coarse_label_names']
    # train 
    data_train = load_chunk('train')
    fine_labels_train = data_train[b'fine_labels']
    coarse_labels_train = data_train[b'coarse_labels']
    train = data_train[b'data']
    # test
    data_test = load_chunk('test')
    fine_labels_test = data_test[b'fine_labels']
    coarse_labels_test = data_test[b'coarse_labels']
    test = data_test[b'data']
    return coarse_labels_train, train, coarse_labels_test, test, coarse_label_names

def show_img(dataset, index):
    x = dataset[index]
    # x : (3073) -> (32 x 32 x 3)
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
    print(coarse_labels_train[index])
    print(np.array(coarse_label_names)[coarse_labels_train[index]])

class SVGLoss:
    def __init__(self) -> None:
        pass

    def forward(self, results, labels): # results : (n, m) , labels : (n,)
        self.results = results # needed for the backprop (this step don't need backprop)
        self.labels = labels # needed for the backprop (this step don't need backprop)
        self.Y = self.results[range(len(self.results)), self.labels] # (n,)
        self.subs = ( np.array([self.Y]).transpose() - results ) + 1 # (n, m)
        self.maxs = np.maximum(0, self.subs) # (n, m)
        self.sum_maxs = np.sum(self.maxs, axis=1, keepdims=True) - 1 # (n,)
        self.loss = np.sum(self.sum_maxs) / (self.results.shape[1] - 1) # INT
        return self.loss

    def backward(self):
        self.dsum_maxs = np.ones_like(self.sum_maxs) * (1 /( self.results.shape[1] - 1) ) # (n,)
        self.dmaxs = np.ones_like(self.maxs) * self.dsum_maxs # (n,m) * (n,) -> (n,m)
        self.dsubs = np.zeros_like(self.subs)
        self.dsubs[np.where(self.subs > 0)] = 1
        self.dsubs *= self.dmaxs # (n, m)
        self.dresults = -np.ones_like(self.results) * self.dsubs # (n,m)
        self.dY = np.sum(np.ones_like(self.results) * self.dsubs, axis=1) # (n,)
        temp = np.zeros_like(self.results) # (n,m)
        temp[range(len(self.results)), self.labels] = 1 # (n,m)
        self.dresults += temp * np.array([self.dY]).transpose() # self.dY se puede usar de esta forma 
        return self.dresults

class SoftmaxLoss:
    def __init__(self) -> None:
        pass

    def forward(self, results, labels, accuracy=False):
        self.labels = np.array(labels) # needed for the backprop
        self.results = results
        self.exps = np.exp(self.results) # (n, m)
        self.sum_exps = np.sum(self.exps, axis=1, keepdims=True) # (n,1)
        self.softmax = self.exps / self.sum_exps  # (n, m)
        if accuracy:
            return self.softmax
        self._loss = -np.log(self.softmax) # (n, m)
        self.loss = np.sum(self._loss[range(len(labels)), self.labels])/len(labels) # int
        return self.loss # int

    def backward(self): # derivada previa (por defecto 1 ya que se supone que es la ultima capa)
        self.d_loss = np.zeros_like(self._loss)
        self.d_loss[range(len(self.labels)), self.labels] = 1/len(self.labels) # (n, m)
        self.d_softmax = (-1/self.softmax) * self.d_loss # (n, m)
        self.dexps = (np.ones_like(self.exps) * (1/self.sum_exps)) * self.d_softmax # (n, m) * (n, 1) -> (n, m)
        self.dsum_exps = np.sum((-self.exps / self.sum_exps ** 2) * self.d_softmax, axis=1, keepdims=True) # (n, 1)
        self.dexps += np.ones_like(self.exps) * self.dsum_exps # (n, m)
        self.dresults = self.exps * self.dexps # (n, m)
        return self.dresults

# ACTIVATION FUNCTIONS
# Sigmoid
# tanh
# ReLU -> max(0, x)
# Leaky ReLU
# Maxout
# ELU
class ReLU:
    def __init__(self) -> None:
        pass

    def forward(self, x, test=False): # x : (n, m)
        self.x = np.array(x) # (n, m)
        self.activation = np.maximum(0, self.x) # (n, m)
        return self.activation 

    def backward(self, prev): # prev == dLoss/d_activation; prev.shape == d_activation.shape
        self.dx = np.zeros_like(self.x) # (n, m)
        self.dx[np.where(self.x > 0)] = 1 # (n, m)
        self.dx *= prev # (n, m)
        return self.dx

class LReLU:
    def __init__(self, alpha=0.001) -> None:
        self.alpha = alpha
        self.dalpha = None

    def forward(self, x): # x : (n, m)
        self.x = np.array(x) # (n, m)
        self.alpha_x = self.alpha * self.x # (n, m)
        self.activation = np.maximun(self.alpha_x, self.x) # (n, m)
        return self.activation 

    def backward(self, prev): # prev == dLoss/d_activation; prev.shape == d_activation.shape
        # when x > 0
        self.dx = np.zeros_like(self.x) # (n, m)
        self.dx[np.where(self.x > 0)] = 1 # (n, m)
        self.dx *= prev # (n, m)
        # when x < 0
        self.dalpha_x = np.zeros_like(self.x) # (n, m)
        self.dalpha_x[np.where(self.x < 0)] = 1 # (n, m)
        self.dalpha_x *= prev # (n, m)
        # 
        self.dx += np.ones_like(self.x) * self.alpha * self.dalpha_x # (n, m)
        self.dalpha = np.sum(self.x * self.dalpha_x) # INT
        return self.dx

    def update(self):
        self.alpha += self.dalpha

class BatchNorm:
    def __init__(self, out_size, e=0.0001) -> None:
        self.alpha = np.ones((1, out_size))
        self.beta = np.zeros((1, out_size))
        self.e = e
        self.means = []
        self.vars = []

    def forward(self, X, test=False): # (n, m) -> n : len_batch, m : input_len
        self.X = X
        if test: # 
            self.mean = np.mean(np.array(self.means), axis=0)
            self.variance = np.mean(np.array(self.vars), axis=0)
        else: # 
            # mini-batch mean
            self.mean = np.sum(self.X, axis=0) / self.X.shape[0] # (m)
            self.means.append(self.mean)
            # mini-batch variance : 1/len_batch*(sum(xi - mean)**2)
            self.rest = self.X - self.mean # (n, m)
            self.rest_pow = self.rest ** 2 # (n, m)
            self.sum_rest_pow = np.sum(self.rest_pow, axis=0) # (m)
            self.variance = self.sum_rest_pow / self.X.shape[0] # (m)
            self.vars.append(self.variance)
        # normalize : (xi - mean)/(variance + e)**-2
        self.sqrt_variance = np.sqrt(self.variance + self.e) # (m)
        self.norm = (self.X - self.mean) /  self.sqrt_variance # (n, m)
        # scale and shift
        self.norm_batch = self.alpha * self.norm + self.beta # (1, m) * (n, m) + (1, m) = (n, m)
        return self.norm_batch

    def backward(self, prev): # (n, m)
        self.dalpha = np.sum(self.norm * prev, axis=0) # (m)
        self.dbeta = np.sum(prev, axis=0) # (m)
        self.dnorm = self.alpha * prev # (n, m)
        self.dX = (1/self.sqrt_variance) * self.dnorm # (n, m)
        self.dmean = np.sum((-1/self.sqrt_variance) * self.dnorm, axis=0) # (m)
        self.dsqrt_variance = np.sum((-(self.X - self.mean) / self.sqrt_variance ** 2) * self.dnorm, axis=0) # (m)
        self.dvariance = 0.5 * (self.variance ** -0.5) * self.dsqrt_variance # (m)
        self.dsum_rest_pow = self.X.shape[0] * self.dvariance # (m)
        self.drest_pow = np.ones_like(self.rest_pow) * self.dsum_rest_pow # (n, m)
        self.drest = 2*(self.rest) * self.drest_pow # (n, m)
        self.dX += np.ones_like(self.X) * self.drest # (n, m)
        self.dmean += -np.sum(self.drest, axis=0) # (m)
        self.dX += np.ones_like(self.X) * 1/self.X.shape[0] * self.dmean # (n, m)
        return self.dX

    def update(self, gradient_decent=0.001):
        self.alpha += -gradient_decent * self.dalpha
        self.beta  += -gradient_decent * self.dbeta


class LinearLayer:
    def __init__(self, input_size, output_size, xavier_init=False, init_factor = 1, name="linear layer", ensembles=False):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.x = None # needed for the backprop
        self.w = np.random.randn(self.input_size, self.output_size) * init_factor
        # Xavier initialization
        if xavier_init:
            self.w = self.w / np.sqrt(input_size / 2) # 
        self.b = np.zeros(self.output_size) # initialize with 0.001 if the next layer is ReLU
        self.xw = None # needed for the backprop
        self.dx = None # 
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.optimizer1 = Optimizer()
        self.optimizer2 = Optimizer()
        self.step = 0
        # Emsembles
        self.ensembles = ensembles
        if self.ensembles:
            self.w_test = self.w
            self.b_test = self.b

    def forward(self, x, test=False):
        if test and self.ensembles:
            self.x = np.array(x) # needed for the back propagation; x = ( any, input_size)
            self.xw = x @ self.w_test + self.b_test  # (any, input_size) @ (input_size, output_size) = (any, output_size)
            return self.xw
        else:
            self.x = np.array(x) # needed for the back propagation; x = ( any, input_size)
            self.xw = x @ self.w + self.b  # (any, input_size) @ (input_size, output_size) = (any, output_size)
            return self.xw

    def backward(self, prev): # prev == dLoss/dxw; prev.shape == xw.shape
        self.dx = prev @ self.w.transpose()  # (any, output_size) @ (output_size, input_size) = (any, input_size)
        self.dw = self.x.transpose() @ prev # (input_size, any) @ (any, output_size) = (input_size, ouput_size)
        self.db = np.sum(prev) # 
        return self.dx
    
    def update(self, lr, optimizer="SGD"):
        # SGD (lr: 0.1 , loss: 2.477, val_acc: 0.2254) (lr: 0.001 , loss: 2.996, val_acc: 0.1253)
        #self.w = self.optimizer1.SGD(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.SGD(self.b, self.db, lr=lr)
        # momentum (lr: 0.1 , loss: 2.567, val_acc: 0.1964) (lr: 0.001 , loss: 2.996, val_acc: 0.1059)
        #self.w = self.optimizer1.momentum(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.momentum(self.b, self.db, lr=lr)
        # ANG (lr: 0.1 , loss: 2.8483 , val_acc: 0.1876)
        #self.w = self.optimizer1.NAG(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.NAG(self.b, self.db, lr=lr)
        # AdaGrad (lr: 0.1, loss: NaN , val_acc: 0.5) (lr: 0.001, loss: 2.523 , val_acc: 0.1996)
        #self.w = self.optimizer1.AdaGrad(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.AdaGrad(self.b, self.db, lr=lr)
        # RMSProp (lr: 0.1 , loss: nan, val_acc: 0.05) (lr: 0.001 , loss: 2.405 , val_acc: 0.2658)
        # (lr: 0.001, ensembles=True, loss: 2.405 , val_acc: 0.2912)
        #self.w = self.optimizer1.RMSProp(self.w, self.dw, lr=lr)
        #self.b = self.optimizer2.RMSProp(self.b, self.db, lr=lr)
        # IF USE ADAM (don't neet learning grade decay)
        # (lr: 0.1 , loss: nan, val_acc: 0.05) (lr: 0.001, loss: 2.989, val_acc: 0.0587)
        # (lr: 0.001, ensembles=True, loss: 2.305 , val_acc: 0.2511)
        self.step += 1
        self.w = self.optimizer1.Adam(self.w, self.dw, lr=lr, t=self.step)
        self.b = self.optimizer2.Adam(self.b, self.db, lr=lr, t=self.step)
        # ENSEMBLES
        if self.ensembles:
            self.w_test = 0.995*self.w_test + 0.005*self.w
            self.b_test = 0.995*self.b_test + 0.005*self.b

class Conv:
    def __init__(self, in_channels, kernel_size: int, out_channels: int, stride = 1):
        """
        filter_size   : dim of filter
        n_filters : number of filters
        """
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.n_kernels = out_channels
        self.kernels = np.random.randn(self.n_kernels, self.kernel_size, self.kernel_size, self.in_channels)
        self.kernels = self.kernels.astype(np.float32)
        self.output = None
        self.stride = stride
        self.input = None
        self.d_k = None
        self.d_x = None

    def forward(self, x): # x : (width, height, deep)
        # save input, needed for backprop
        self.input = x
        # calculate the output size
        if self.output is None:
            width, height, deep = x.shape
            output_width = int(((width - self.kernel_size)/self.stride) + 1)
            self.output = np.zeros((output_width, output_width, self.n_kernels)) # (output_width, output_height, out_channels)
        # one section convolution function
        def conv(img_sec_pos, kernels):
            _x, _y = img_sec_pos
            img_sec = x[_x:_x+self.kernel_size, _y:_y+self.kernel_size, :]
            img_sec = img_sec.astype(np.float32)
            # img_sec : (3, 3, 3) , kernels : (5, 3, 3, 3)
            result_conv = np.array([np.sum(kernel_result) for kernel_result in img_sec * kernels])
            return result_conv
        # doing convolutions
        for i_width in range(self.output.shape[0]):
            for i_height in range(self.output.shape[1]):
                self.output[i_width][i_height] = conv((i_width, i_height), self.kernels)
        return self.output # (output_width, output_height, out_channels)

    def backward(self, prev): # prev : (output_width, output_height, out_channels)
        # one section convolution function
        def conv(img_sec_pos, kernels):
            _x, _y = img_sec_pos
            img_sec = self.input[_x:_x+kernels.shape[0], _y:_y+kernels.shape[1], :]
            # img_sec : (3, 3, 3) , kernels : (5, 3, 3, 3)
            kernel_results = np.array([(img_sec * np.reshape(kernels, (kernels.shape[0], kernels.shape[0], 1)))[:,:,deep] for deep in range(self.input.shape[2])])
            result_conv = np.array([np.sum(kernel_result) for kernel_result in kernel_results]) # (img_width, img_height, deep) * (output_width, output_height)
            return result_conv
        # d_k backprop
        self.d_k = np.zeros_like(self.kernels) # kernels : (n_kernels, kernel_size, kernel_size, in_channels)
        for act in range(self.output.shape[2]): # activation
            for x in range(self.d_k.shape[1]): # row
                for j in range(self.d_k.shape[2]): # col
                    self.d_k[act][x][j] = conv((x,j), prev[:,:,act])
        # d_x backprop
        self.d_x = np.zeros_like(self.input) # e.g. (6, 6, 3)
        temp_d_x = np.array([np.zeros_like(self.input) for _ in range(self.kernels.shape[0])]) # creamos una pila de vectores de dimension d_x, cada uno de los vectores contendra la derivada con respecto a una activacion, al final se sumara para solo tener un vector de dimension d_x
        # temp_d_x : e.g. (5, 6, 6, 3)
        for i_kernel in range(self.kernels.shape[0]):
            for deep in range(self.kernels.shape[3]):
                for row in range(self.kernels.shape[1]):
                    for col in range(self.kernels.shape[2]):
                        #temp_d_x[i_kernel,row:row+prev.shape[0],col:col+prev.shape[1],deep] += self.kernels[i_kernel][row][col][deep] * prev[:,:,i_kernel]
                        temp_d_x[i_kernel,row:row+prev.shape[0],col:col+prev.shape[1],deep] += self.kernels[i_kernel,row,col,deep] * prev[:,:,i_kernel]
                        #print(temp_d_x)
        self.d_x = np.sum(temp_d_x, axis=0)
        return self.d_x

class Dropout:
    def __init__(self) -> None:
        pass

    # x : input (batch)
    # p : percent of dropout
    def forward(self, x, p=0.5, test=False):
        if test:
            x *= p # (n, m)
            return x
        else:
            self.U = np.random.rand(*x.shape) < p # (n, m)
            x *= self.U # (n, m)
            return x

    def backward(self, prev): # x.shape == prev.shape
        self.dx = self.U * prev # (n, m)
        return self.dx

class Optimizer:
    def __init__(self) -> None:
        # momentum
        self.momentum_v = None
        # Nesterov Accelerated Gradient (NAG)
        self.nesterov_v = None
        # AdaGrad
        self.AdaGrad_cache = None
        # RMSProp
        self.rms_cache = None
        # Adam ()
        self.adam_m = None
        self.adam_v = None

    def SGD(self, w, dw, lr=0.001):
        w += -lr * dw
        return w

    def momentum(self, w, dw, lr=0.001, mu=0.5): # commor mu in range [~0.5, 0.99]
        if self.momentum_v is None:
            self.momentum_v = np.zeros_like(w)
        self.momentum_v = mu * self.momentum_v - lr * dw
        w += self.momentum_v
        return w
 
    def NAG(self, w, dw, lr=0.001, mu=0.5): # Nesterov Accelerated Gradient
        if self.nesterov_v is None:
            self.nesterov_v = np.zeros_like(w)
        v_prev = self.nesterov_v
        self.nesterov_v = mu * self.nesterov_v - lr * dw
        w += -mu * v_prev + (1 + mu) * self.nesterov_v
        return w

    def AdaGrad(self, w, dw, lr=0.001): # Ada Grad
        if self.AdaGrad_cache is None:
            self.AdaGrad_cache = np.zeros_like(w)
        self.AdaGrad_cache += dw**2
        w += -lr * dw / (np.sqrt(self.AdaGrad_cache) + 1e-7)
        return w

    def RMSProp(self, w, dw, lr=0.001, decay_rate=0.99): # 
        if self.rms_cache is None:
            self.rms_cache = np.zeros_like(w)
        self.rms_cache = decay_rate * self.rms_cache + (1 - decay_rate) * dw**2
        w += -lr * dw / (np.sqrt(self.rms_cache) + 1e-7)
        return w
    
    def Adam(self, w, dw, lr=0.001, beta1=0.9, beta2=0.999, t=1): # 
        if self.adam_m is None:
            self.adam_m = np.zeros_like(w)
        if self.adam_v is None:
            self.adam_v = np.zeros_like(w)
        self.adam_m = beta1*self.adam_m + (1-beta1)*dw
        self.adam_v = beta2*self.adam_v + (1-beta2)*(dw**2)
        adam_m = self.adam_m / (1-beta1**t)
        adam_v = self.adam_v / (1-beta2**t)
        w += -lr*adam_m/(np.sqrt(adam_v)+1e-7)
        return w

class Model: # arquitectura de la NN
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
