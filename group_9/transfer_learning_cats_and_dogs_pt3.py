import os
import random

import dlvc.batches as batches
import dlvc.datasets.pets as datasets
from dlvc.dataset import Subset
import dlvc.ops as ops
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

input_shape = (0, 3, 224, 224)
dataset_path = os.path.join(os.pardir,
                            "cifar-10-batches-py")  # assumes that CIFAR-10 dataset is placed in parent folder
model_filepath = os.path.join(os.curdir, f"best_model_transfer_learning.pt")

# used for per-channel normalization (calculated in calc_training_set_stats.py)
imagenet_stats = {
    # Statistics of ImageNet (rescaled to [0 ,255])
    # Source: https://pytorch.org/vision/stable/models.html (Accessed: 2022-05-20)
    "mean": np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32),
    "std": np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
}


def seed(seed_value=29):
    """
    Reset all seeds to ensure reproducibility
    Source: https://pytorch.org/docs/stable/notes/randomness.html (Accessed: 2022-04-11)
    """
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    val_acc_best = Accuracy()
    early_stop_cnt = 0
    for e in range(max_epochs):
        print(f"Epoch {e + 1}")

        train_loss = []
        for train_batch in train_batches:
            data = train_batch.data
            labels = train_batch.label
            train_loss.append(clf.train(data, labels))

        train_loss_np = np.array(train_loss)

        print(f"train loss: {np.mean(train_loss_np):.3f} ± {np.std(train_loss_np):.3f}")

        val_acc_curr = Accuracy()
        for val_batch in val_batches:
            data = val_batch.data
            label = val_batch.label
            output = clf.predict(data)
            val_acc_curr.update(output, label)

        if val_acc_curr > val_acc_best:
            val_acc_best = val_acc_curr
            torch.save(net.state_dict(), model_filepath)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        print(val_acc_curr)

        if early_stop_cnt == epochs_early_stop:
            print(f"Early stopping triggered!")
            break

    print(f"--------------------\nval acc (best): {val_acc_best.accuracy():.3f}")


def test():
    net.load_state_dict(torch.load(os.path.join(os.curdir, model_filepath)))

    test_acc = Accuracy()
    with torch.no_grad():
        for test_batch in test_batches:
            data = test_batch.data
            label = test_batch.label
            output = clf.predict(data)
            test_acc.update(output, label)

    print(f"test acc {test_acc.accuracy():.3f}")

# (hyper)parameters
max_epochs = 30
epochs_early_stop = 5
batch_size = 128
learning_rate = 1e-3
weight_decay = 0.

# reset seeds
seed()

train_ops_list = [ops.resize(input_shape[2:]), ops.type_cast(np.float32), ops.normalizePerChannel(imagenet_stats["mean"], imagenet_stats["std"])]

test_val_ops_list = [ops.resize(input_shape[2:]), ops.type_cast(np.float32), ops.normalizePerChannel(imagenet_stats["mean"], imagenet_stats["std"]), ops.hwc2chw()]
test_val_ops = ops.chain(test_val_ops_list)

# load training and validation batches
train_set = datasets.PetsDataset(dataset_path, Subset.TRAINING)
train_ops_list += [ops.hwc2chw()]
train_batches = batches.BatchGenerator(train_set, batch_size, True, ops.chain(train_ops_list))

val_set = datasets.PetsDataset(dataset_path, Subset.VALIDATION)
val_batches = batches.BatchGenerator(val_set, batch_size, False, test_val_ops)

# create network, freeze all layers except fully connected layer and wrap in CnnClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CUDA, if available
num_classes = train_set.num_classes()

net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, num_classes)  # replace classification layer
net.to(device)
clf = CnnClassifier(net, input_shape=input_shape, num_classes=num_classes, lr=learning_rate,
                    wd=weight_decay)

# start training procedure
train()

# load test batches
test_set = datasets.PetsDataset(dataset_path, Subset.TEST)
test_batches = batches.BatchGenerator(test_set, batch_size, False, test_val_ops)

# start testing procedure
test()
