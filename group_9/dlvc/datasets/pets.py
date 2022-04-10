from ..dataset import Sample, Subset, ClassificationDataset

import os
import pickle

import numpy as np


def unpickle(file):
    try:
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')
    except FileNotFoundError:
        raise ValueError(f"{file} cannot be found")


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    subset_to_files = {
        Subset.TRAINING: ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'],
        Subset.VALIDATION: ['data_batch_5'],
        Subset.TEST: ['test_batch']
    }

    labels = [0, 1]  # cat, dog

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
        meta_label_names = unpickle(os.path.join(fdir, 'batches.meta'))[b'label_names']
        orig_label_cat = meta_label_names.index(b'cat')
        orig_label_dog = meta_label_names.index(b'dog')

        data = []
        self.idx_to_label = []
        for file in PetsDataset.subset_to_files[subset]:
            b = unpickle(os.path.join(fdir, file))

            orig_labels = np.array(b[b'labels'])
            cat_dog_mask = (orig_labels == orig_label_cat) | (orig_labels == orig_label_dog)
            orig_labels_masked = orig_labels[cat_dog_mask]

            self.idx_to_label += np.select([orig_labels_masked == orig_label_cat, orig_labels_masked == orig_label_dog],
                                           PetsDataset.labels, orig_labels_masked).tolist()
            data.append(b[b'data'][cat_dog_mask].astype(np.uint8))

        self.data = np.reshape(np.vstack(data), (-1, 3, 32, 32)).transpose((0, 2, 3, 1))[:, :, :, ::-1]  # convert images to BGR format
        self._num_classes = len(set(self.idx_to_label))

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

        return Sample(idx, self.data[idx], self.idx_to_label[idx])

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        return self._num_classes
