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
