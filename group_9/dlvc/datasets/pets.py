
from ..dataset import Sample, Subset, ClassificationDataset
import pandas as pd
import numpy as np
import os
##needed to use the CIFAR-10 df

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict






CAT=0
DOG=1

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

        # TODO implement
        ##VALUE ERRORS
        if not os.path.exists(fdir):
            raise ValueError('"{}" cannot be found'.format(fdir))
            
        path = os.path.join(fdir, 'batches.meta')
        if not os.path.exists(path):
            raise ValueError('"{}" cannot be found'.format(path))
            
            
        #batches.meta are the label names
        #get cat and dog index of labels in the batches
        meta_batches = unpickle(path)
        labels = meta_batches[b'label_names']
        index_cat = labels.index(b'cat')
        index_dog = labels.index(b'dog')            
            
        ## read in batch files
        ## 10.000 images per batch, each having 1000 images per class
        ## 50.000 in training batches
        ## 10.000 in test
        ## meaning: get all images with cat and dog label from the batches and we end up
        ## with 10.000 (5.000 x 2) 


        ## use of dir for subsets mapping?

        ## for loop through files to label images?
        ##img as numpy arrays in bgr order
        pass

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # TODO implement

        pass

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds. Negative indices are not supported.
        '''

        # TODO implement

        pass

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # TODO implement

        pass
