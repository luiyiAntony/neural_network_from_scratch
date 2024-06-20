import numpy as np
import pandas as pd
import gzip

class FashionMNIST:
    def __init__(self, root="", train=True):
        if train:
            self.X, self.y = self.load_train()
        else:
            self.X, self.y = self.load_test()
        self.data = list(zip(self.X, self.y))
        #self.X = self.X.reshape(self.X.shape[0], -1)
        #self.label_names = self.load_labels()

    def load_train(self):
        # DATA
        dir = 'data/mnist/'
        #dir = '/content/drive/MyDrive/Deep_Learning/data/mnist/'
        fdata = gzip.open(dir + 'train-images-idx3-ubyte.gz', 'r')
        # TARGETS
        ftargets = gzip.open(dir + 'train-labels-idx1-ubyte.gz', 'r')

        image_size = 28
        num_images = 60000

        fdata.read(16)
        ftargets.read(8)
        bufdata = fdata.read(image_size * image_size * num_images)
        buftargets = ftargets.read(num_images)
        data = np.frombuffer(bufdata, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        targets = np.frombuffer(buftargets, dtype=np.uint8).astype(np.int64)
        return data, targets

    def load_test(self):
        # DATA
        dir = 'data/mnist/'
        #dir = '/content/drive/MyDrive/Deep_Learning/data/mnist/'
        fdata = gzip.open(dir + 't10k-images-idx3-ubyte.gz', 'r')
        # TARGETS
        ftargets = gzip.open(dir + 't10k-labels-idx1-ubyte.gz', 'r')

        image_size = 28
        num_images = 10000

        fdata.read(16)
        ftargets.read(8)
        bufdata = fdata.read(image_size * image_size * num_images)
        buftargets = ftargets.read(num_images)
        data = np.frombuffer(bufdata, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        targets = np.frombuffer(buftargets, dtype=np.uint8).astype(np.int64)
        return data, targets

class CIFAR10:
    """
    PARAMS:
        train: 
            TRUE to get the train dataset
            FALSE to get the test dataset
        visualization:
            TRUE to reshape the images for visualization 
            FALSE to reshape the images for convolution 
            (NOTE: read description below for more detail)

    DESCRIPTION:

    Dataset size :
        TRAIN:  (50000, 3072)
        TEST:   (10000, 3072)
    Each row is an image:
        red     = row[   0 : 1024] # red values
        green   = row[1024 : 2048] # green values
        blue    = row[2048 : 3072] # blue values

    Reshape for visualization:

        Each column make a RGB pixel: 

            (red[i], green[i], blue[i]) -> (1024, 3)

        then reshape: 

            img_shape           = (1024, 3) -> (32, 32, 3)
            train_dataset_shape = (50000, 32, 32, 3)
            test_dataset_shape  = (10000, 32, 32, 3)

    Reshape for convolution:

        Generate an 32 x 32 image for each channel

            red     = row[   0 : 1024] reshape to (32, 32)
            green   = row[1024 : 2048] reshape to (32, 32)
            blue    = row[2048 : 3072] reshape to (32, 32)

        then:
            img_shape           = (3, 32, 32)
            train_dataset_shape = (50000, 3, 32, 32)
            test_dataset_shape  = (10000, 3, 32, 32)
    """
    def __init__(self, root="", train=True, visualization=False):
        self.visualization = visualization
        if train:
            self.X, self.y = self.load_train()
        else:
            self.X, self.y = self.load_test()
        self.X = self.preproces_data(self.X)
        self.data = list(zip(self.X, self.y))
        self.label_names = self.load_labels()

    def preproces_data(self, X):
        """
        Convert (len(X), 3072) -> (len(X), 32, 32, 3)
        Convert (len(X), 3072) -> (len(X), 32, 32, 3)
        """
        reshape_X = np.reshape(X, (X.shape[0], 3, 1024)).astype(np.float64)
        if (not self.visualization):
            reshape_X = np.reshape(reshape_X, (X.shape[0], 3, 32, 32))
            return reshape_X
        reshape_X = np.array([np.reshape(np.ravel(item, order='F'), (1024, 3)) for item in reshape_X])
        reshape_X = np.reshape(reshape_X, (X.shape[0], 32, 32, 3))
        return reshape_X

    def load_train(self):
        data_train = self.load_chunk('train')
        fine_labels_train = data_train[b'fine_labels']
        coarse_labels_train = data_train[b'coarse_labels']
        train = data_train[b'data']
        train = np.array(train)
        coarse_labels_train = np.array(coarse_labels_train)
        return train, coarse_labels_train

    def load_test(self):
        # test
        data_test = self.load_chunk('test')
        fine_labels_test = data_test[b'fine_labels']
        coarse_labels_test = data_test[b'coarse_labels']
        test = data_test[b'data']
        test = np.array(test)
        coarse_labels_test = np.array(coarse_labels_test)
        return test, coarse_labels_test

    def load_labels(self):
        # label's name
        meta = self.load_chunk('meta')
        fine_label_names = meta[b'fine_label_names']
        coarse_label_names = meta[b'coarse_label_names']
        return coarse_label_names

    def load_chunk(self, file_name):
        import pickle
        #data_dir = '/content/drive/MyDrive/Deep_Learning/data/'
        data_dir = 'data/'
        with open(data_dir + file_name, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        # meta: fine_label_names, coarse_label_names
        # train = test: filenames, batch_label, fine_labels, coarse_labels, data
        #for i, d in enumerate(dict):
            #print(d)
            #print(d, dict[d])
        return dict

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        """
        Initialize the DataLoader object.
        
        :param data: The dataset to load, expected to be a numpy array.
        :param batch_size: The number of samples per batch.
        :param shuffle: Whether to shuffle the data at the start of each epoch.
        """
        self.data = data # (X, y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data.X))
        self.on_epoch_start()

    def on_epoch_start(self):
        """
        Shuffle the data at the start of each epoch if shuffle is True.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        """
        Number of batches per epoch.
        
        :return: Number of batches per epoch.
        """
        return int(np.ceil(len(self.data.X) / self.batch_size))
    
    def __iter__(self):
        """
        Create an iterator that yields batches of data.
        
        :return: Iterator for the DataLoader.
        """
        for start_idx in range(0, len(self.data.X), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.data.X))
            batch_indices = self.indices[start_idx:end_idx]
            #yield self.data[batch_indices,0:-1], self.data[batch_indices, self.data.shape[1] - 1]
            yield self.data.X[batch_indices], self.data.y[batch_indices]
    
    def __getitem__(self, index):
        """
        Get a batch of data by index.
        
        :param index: Index of the batch to retrieve.
        :return: Batch of data as a numpy array.
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data.X))
        batch_indices = self.indices[start_idx:end_idx]
        return self.data.X[batch_indices], self.data.y[batch_indices]
    











