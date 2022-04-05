def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label = unpickle('C:\\Users\\kinos\\Documents\\TU_Wien\\Deeplearning\\dlvc_ss22-main\\assignments\\Code\\dlvc\\datasets\\cifar-10-batches-py\\data_batch_1')
print(label)
