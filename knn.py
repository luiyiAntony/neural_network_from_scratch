import numpy as np

# 5 train images (2x2 size)
train_imgs = [
    [ 1, 2, 3, 4],
    [ 4, 5, 6, 7],
    [ 8, 9,10,11],
    [12,13,14,15],
    [16,17,18,19],
]

# 2 test images (2x2 size)
test_imgs = [
    [20,21,22,23],
    [24,25,26,27],
]

# HYPERPARAMS
K = 3
#distance_metric = "manhattan_distance" # "euclidean"
distance_metric = "euclidean"

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
    def __init__(self):
        pass

    def train(self, dataSet, labels, k):
        # dataset (num_train, size_img)
        # labels (num_train)
        self.x = np.array(dataSet)
        self.y = np.array(labels)
        self.K = k

    def predict(self, x, metric='manhattan_distance'):
        """
        x : test images set (num_test, size_img)
        """
        def manhattan_distance(x):
            x = np.array(x)
            predictions = np.zeros(x.shape[0], dtype = self.y.dtype)
            for i in range(len(x)):
                distances = np.sum(np.abs(self.x - x[i]), axis=1)
                k_mins = np.zeros(K, dtype=self.y.dtype)
                for j in range(self.K):
                    idx_min = np.argmin(distances)
                    k_mins[j] = self.y[idx_min]
                    np.delete(distances, idx_min)
                unique, counts = np.unique(k_mins)
                predictions[i] = unique[np.argmax(counts)]
                print(f"img: {i}, label: {predictions[i]}, distance: {distances[idx_min]}")
            return predictions

        def euclidean_distance(x):
            x = np.array(x)
            predictions = np.zeros(x.shape[0], dtype = self.y.dtype)
            for i in range(len(x)):
                distances = np.sum((self.x - x[i]) ** 2, axis=1) ** 0.5
                idx_min = np.argmin(distances)
                predictions[i] = self.y[idx_min]
                print(f"img: {i}, label: {predictions[i]}, distance: {distances[idx_min]}")
            return predictions
        
        # classified_images : (num_test)
        if metric == 'manhattan_distance':
            classified_imgs = manhattan_distance(x)
        else:
            classified_imgs = euclidean_distance(x)
        return classified_imgs

if __name__ == "__main__":
    # load data
    fine_labels_train, train, fine_labels_test, test, coarse_label_names = load_data()
    #test = np.random.randint(256, size=(10000, 3072))
    # train
    knn = NearestNeighbor()
    knn.train(train, fine_labels_train, k=5)
    # test
    # generate 20 random indexs between 0 to 9999
    test_indexs = np.random.randint(0, 9999, size=20)
    classified_imgs = knn.predict(test[test_indexs], metric='manhattan_distances')
    difference = np.array(np.array(fine_labels_test)[test_indexs]) == classified_imgs
    print(difference)
    print(difference.sum())