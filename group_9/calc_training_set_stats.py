# Calculates the mean and standard deviation of the training for per channel normalization
# Since the dataset is small we can calculate the statistics in-memory using a single batch
import os

import dlvc.batches as batches
import dlvc.datasets.pets as datasets
from dlvc.dataset import Subset
import dlvc.ops as ops

import numpy as np

# put all images in one batch (assumes that dataset is stored in parent directory)
train_set = datasets.PetsDataset(os.path.join(os.pardir, "cifar-10-batches-py"), Subset.TRAINING)
train_batches = batches.BatchGenerator(train_set, len(train_set), False, ops.chain([ops.type_cast(np.float32)]))

# get the data of the single batch (N, H, W, C) and calculate statistics
data = next(b for b in train_batches).data
mean = np.mean(data, axis=(0, 1, 2))
std = np.std(data, axis=(0, 1, 2))

print(f"{mean=}, {std=}") # mean = [105.653984, 117.07664 , 126.680565], std = [64.33777 , 63.024414, 64.452705]
