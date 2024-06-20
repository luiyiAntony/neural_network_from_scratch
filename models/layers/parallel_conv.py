import cupy as cp
from models.layers.layer import Module

class ConvLayer(Module):
    def __init__(self, num_filters, filter_size, input_shape, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride
        self.padding = padding

        # Initialize filters and biases
        # self.filters = cp.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        # self.biases = cp.zeros((num_filters, 1))
        self.params['W'] = cp.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.params['b'] = cp.zeros((num_filters, 1))
        self.grads['W'] = cp.zeros_like(self.params['W'])
        self.grads['b'] = cp.zeros_like(self.params['b'])

    def forward(self, X):
        print("FORWARD...\n")
        self.X = cp.asarray(X)
        X = cp.asarray(X)
        batch_size, in_depth, in_height, in_width = X.shape

        out_height = (in_height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.filter_size + 2 * self.padding) // self.stride + 1

        # Initialize output
        out = cp.zeros((batch_size, self.num_filters, out_height, out_width))

        # Apply padding
        X_padded = cp.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size

                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.num_filters):
                    out[:, k, i, j] = cp.sum(X_slice * self.params['W'][k, :, :, :], axis=(1, 2, 3))

        # Adjust biases shape for broadcasting
        return cp.asnumpy(out + self.params['b'].reshape(1, self.num_filters, 1, 1))

    # def backward(self, d_out, learning_rate):
    def backward(self, d_out):
        print("BACKWARD...\n")
        d_out = cp.asarray(d_out)
        batch_size, in_depth, in_height, in_width = self.X.shape
        _, _, out_height, out_width = d_out.shape

        # Initialize gradients
        dX = cp.zeros_like(self.X)
        self.grads['W'] = cp.zeros_like(self.params['W'])
        self.grads['b'] = cp.zeros_like(self.params['b'])

        X_padded = cp.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        dX_padded = cp.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size

                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.num_filters):
                    self.grads['W'][k, :, :, :] += cp.sum(X_slice * d_out[:, k, i, j][:, None, None, None], axis=0)
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += self.params['W'][k, :, :, :] * d_out[:, k, i, j][:, None, None, None]

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grads['b'] = cp.sum(d_out, axis=(0, 2, 3)).reshape(self.num_filters, 1)

        # Update weights and biases
        # self.filters -= learning_rate * dW
        # self.biases -= learning_rate * db

        return cp.asnumpy(dX)
