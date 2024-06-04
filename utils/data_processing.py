import numpy as np
import pandas as pd
import gzip

class FashionMNIST:
    def __init__(self, root="", train=True):
        if train:
            self.X, self.y = self.load_train()
        else:
            self.X, self.y = self.load_test()
        self.X = self.X.reshape(self.X.shape[0], -1)
        #self.label_names = self.load_labels()

    def load_train(self):
        # DATA
        dir = 'data/mnist/'
        #dir = 'data/FashionMNIST/raw/'
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
        #dir = 'data/FashionMNIST/raw/'
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
    def __init__(self, root="", train=True):
        if train:
            self.X, self.y = self.load_train()
        else:
            self.X, self.y = self.load_test()
        self.label_names = self.load_labels()

    def load_train(self):
        data_train = self.load_chunk('train')
        fine_labels_train = data_train[b'fine_labels']
        coarse_labels_train = data_train[b'coarse_labels']
        train = data_train[b'data']
        return train, coarse_labels_train

    def load_test(self):
        # test
        data_test = self.load_chunk('test')
        fine_labels_test = data_test[b'fine_labels']
        coarse_labels_test = data_test[b'coarse_labels']
        test = data_test[b'data']
        return test, coarse_labels_test

    def load_labels(self):
        # label's name
        meta = self.load_chunk('meta')
        fine_label_names = meta[b'fine_label_names']
        coarse_label_names = meta[b'coarse_label_names']
        return coarse_label_names

    def load_chunk(self, file_name):
        import pickle
        #data_dir = 'drive/MyDrive/Deep_Learning/data/'
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
        
        :param data: The dataset to load, expected to be a pandas DataFrame.
        :param batch_size: The number of samples per batch.
        :param shuffle: Whether to shuffle the data at the start of each epoch.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
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
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __iter__(self):
        """
        Create an iterator that yields batches of data.
        
        :return: Iterator for the DataLoader.
        """
        for start_idx in range(0, len(self.data), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.data))
            batch_indices = self.indices[start_idx:end_idx]
            yield self.data.iloc[batch_indices,0:-1], self.data.iloc[batch_indices, self.data.shape[1] - 1]
    
    def __getitem__(self, index):
        """
        Get a batch of data by index.
        
        :param index: Index of the batch to retrieve.
        :return: Batch of data as a pandas DataFrame.
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))
        batch_indices = self.indices[start_idx:end_idx]
        return self.data.iloc[batch_indices, 0:-1], self.data.iloc[batch_indices, -1]
    











