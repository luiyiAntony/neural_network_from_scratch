import numpy as np

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
    return fine_labels_train, train, fine_labels_test, test, coarse_label_names

class NearestNeighbor:
    """
    Modelo que consiste en almacenar todo el dataset de entrenamiento con sus respectivos labels,
    Para hacer una prediccion recurre a su conocimiento (dataset almacenado),
    calcula la distancia vectorial entre el input y cada uno de sus ejemplos del dataset y busca 
    el que tenga la menor distancia.
    """
    def __init__(self):
        pass

    def train(self, dataSet, labels):
        """
        Almacena todo el dataset de entrenamiento con sus respectivos labels,
        """
        # dataset (num_train, size_img) i.g. (1000, 3072)
        # labels (num_train) i.g. (1000,)
        self.x = np.array(dataSet)
        self.y = np.array(labels)

    def predict(self, x, distance='L1'):
        """
        Para hacer una prediccion recurre a su conocimiento (dataset almacenado),
        calcula la distancia vectorial entre el input y cada uno de sus ejemplos del dataset y busca 
        el que tenga la menor distancia.
        x : test images set (num_test, size_img) (i.g,) -> (20, 3072)
        distance : L1, L2
        """
        num_test = x.shape[0]
        Y_pred = np.zeros(num_test, dtype=self.y.dtype) # ensure the prediction array dimension

        for i in range(num_test):
            # L1 distance
            if distance == "L1":
                distances = np.sum(np.abs(self.x - x[i,:]), axis=1) # get the distances between i-th input image and every image in the model (self.Xtr)
            # L2 distance
            elif distance == "L2":
                distances = np.sqrt(np.sum(np.square(self.x - x[i,:]), axis=1))# get the distances between i-th input image and every image in the model (self.Xtr)
            else:
                raise TypeError("wrong distance argument, choose between L1 and L2")
            min_index = np.argmin(distances) # get the index of the small sum
            Y_pred[i] = self.y[min_index] # predic the label  the 
        return Y_pred

class KNearestNeighbor(object):
    """
    Modelo que consiste en almacenar todo el dataset de entrenamiento con sus respectivos labels,
    Para hacer una prediccion recurre a su conocimiento (dataset almacenado),
    calcula la distancia vectorial entre el input y cada uno de sus ejemplos del dataset y busca 
    k objatos que tengan la menor distancia.
    """
    def __init__(self) -> None:
        pass

    def train(self, X, y, k=3):
        """
        Almacena todo el dataset de entrenamiento con sus respectivos labels,
        """
        self.Xtr = np.array(X)
        self.ytr = np.array(y)
        self.k = k

    def predict(self, x, distance='L1', verbose=False):
        """
        A generalization of NearestNeighbor class
        Para hacer una prediccion recurre a su conocimiento (dataset almacenado),
        calcula la distancia vectorial entre el input y cada uno de sus ejemplos del dataset 
        y busca los k objetos tengan la menor distancia.
        x : test images set (num_test, size_img) (i.g,) -> (20, 3072)
        distance : L1, L2
        """
        n_predictions = x.shape[0]
        Y_pred = np.zeros(n_predictions, dtype=self.ytr.dtype)
        for i in range(n_predictions):
            # distances with L1 and L2
            if distance == "L1":
                distances = np.sum(np.abs(self.Xtr - x[i,:]), axis=1)
            elif distance == "L2":
                distances = np.sqrt(np.sum(np.square(self.Xtr - x[i,:]), axis=1))
            else:
                raise TypeError("wrong distance argument, choose between L1 and L2")
            # calcular los k vecinos
            k_nearest = np.zeros(self.k, dtype=self.ytr.dtype)
            for j in range(self.k):
                idx_min = np.argmin(distances)
                distances[idx_min] = float('inf')
                k_nearest[j] = self.ytr[idx_min]
            # calcular la clase que mas predomina en el grupo de vecinos 
            uniques = np.unique(k_nearest) # guardamos solo los valores unicos de los vecinos para luego contarlos
            uniques_count = [(unique, np.count_nonzero(k_nearest == unique)) for unique in uniques] # (clase, counts) | contamos cuantas vecinos de cada clase existen entre los k vecinos mas cercanos
            nearest_neighbor = max(uniques_count, key= lambda x: x[1])[0] # calculamos la clase que mas predomina en el conjunto de k vecinos mas cercanos
            # apilar al Y_pred
            Y_pred[i] = nearest_neighbor
            # show verbose 
            if verbose:
                print(f"")
        return Y_pred

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # (a - b) ** 2 -> a**2 - 2*a*b + b**2
        _x = np.sum(X**2, axis=1)
        _y = np.sum(self.Xtr**2, axis=1)
        _2xy = -2 * X @  self.Xtr.T
        
        dists = np.sqrt(np.expand_dims(_x, axis=1) + _2xy + np.expand_dims(_y, axis=0))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # match distances with indexes
            val_with_index = [(val, index) for index, val in enumerate(dists[i])]
            # order using the values
            val_with_index.sort(key=lambda pair: pair[0])
            # get only the indexes (first k indexes)
            ordered_neighbors = [pair[1] for pair in val_with_index][:k]
            # get the labels
            clasest_y = self.ytr[ordered_neighbors]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            unique, counts = np.unique(clasest_y, return_counts=True)
            pairs = list(zip(unique, counts))
            pairs.sort(key=lambda pair: pair[1], reverse=True)
            y_pred[i] = pairs[0][0]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

if __name__ == "__main__":
    # load data
    fine_labels_train, train, fine_labels_test, test, coarse_label_names = load_data()
    #test = np.random.randint(256, size=(10000, 3072))

    # create model
    #model = NearestNeighbor()
    model = KNearestNeighbor()

    # train
    #model.train(train, fine_labels_train)
    model.train(train, fine_labels_train, k=5)

    # test 1
    # generate 20 random indexs between 0 to 9999 for get random images from the test dataset
    #test_indexs = np.random.randint(0, 9999, size=100)
    #classified_imgs = model.predict(test[test_indexs], distance='L2')
    #difference = np.array(np.array(fine_labels_test)[test_indexs]) == classified_imgs

    
    # test 2
    # generate 20 random indexs between 0 to 9999 for get random images from the test dataset
    test_indexs = np.random.randint(0, 9999, size=100)
    dists = model.compute_distances_no_loops(test[test_indexs])
    classified_imgs = model.predict_labels(dists)
    difference = np.array(np.array(fine_labels_test)[test_indexs]) == classified_imgs

    # show results
    print(difference)
    print(difference.sum())
