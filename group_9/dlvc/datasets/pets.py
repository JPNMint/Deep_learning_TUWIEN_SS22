from ..dataset import Sample, Subset, ClassificationDataset

import numpy as np
import os
import pickle


def unpickle(file):
    try:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
    except FileNotFoundError:
        raise ValueError('"{}" cannot be found'.format(file))

    return dict


subset_to_batch = {
    Subset.TRAINING: ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'],
    Subset.VALIDATION: ['data_batch_5'],
    Subset.TEST: ['test_batch']
}


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in BGR channel order.
        '''

        # TODO: maybe refactor?
        path = os.path.join(fdir, 'batches.meta')

        meta_batches = unpickle(path)
        labels = meta_batches[b'label_names']
        index_cat = labels.index(b'cat')
        index_dog = labels.index(b'dog')

        self.idx_to_label = []
        self.imgs = np.empty((0, 3072), dtype=np.uint8)
        for f in subset_to_batch[subset]:
            b = unpickle(os.path.join(fdir, f))
            labels = np.array(b[b'labels'])

            self.idx_to_label += [0 if l == index_cat else 1 for l in
                                  (labels[(labels == index_cat) | (labels == index_dog)]).tolist()]

            self.imgs = np.concatenate((self.imgs, b[b'data'][(labels == index_cat) | (labels == index_dog)]))

        self.num_classes = len(set(self.idx_to_label))

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return len(self.idx_to_label)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        if idx < 0:
            raise IndexError("Negative indices are not supported")

        return Sample(idx, np.reshape(self.imgs[idx], (3, 32, 32)).transpose((1, 2, 0))[:, :, ::-1],
                      self.idx_to_label[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return self.num_classes
