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

