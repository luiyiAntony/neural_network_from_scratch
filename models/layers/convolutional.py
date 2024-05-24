# models/layers/convolutional.py
"""
# torch example
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv(x)
"""

class ConvLayer:
    def __init__(self, in_channels, kernel_size: int, out_channels: int, stride = 1):
        """
        in_channels : depth input images
        kernel_size : kernel size (only one because the kernel is square so width and height are equal)
        out_channels : number of kernes (it will be the depth for the output 3d matrix of this convolutional layer)
        stride : size of the steps that the kernel will take
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
