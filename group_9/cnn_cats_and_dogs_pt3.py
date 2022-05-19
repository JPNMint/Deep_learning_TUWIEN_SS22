import argparse
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

input_shape = (0, 3, 32, 32)
dataset_path = os.path.join(os.pardir,
                            "cifar-10-batches-py")  # assumes that CIFAR-10 dataset is placed in parent folder
model_filepath = os.path.join(os.curdir, f"best_model.pt")

# used for per-channel normalization (calculated in calc_training_set_stats.py)
training_set_stats = {
    "mean": np.array([105.653984, 117.07664, 126.680565], dtype=np.float32),
    "std": np.array([64.33777, 63.024414, 64.452705], dtype=np.float32)
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


class PerrosYGatosNet(nn.Module):
    """
    Our own CNN architecture -- Perros-y-Gatos-Net -- a CNN for classifying cats and dogs images.
    Enhanced with Dropout layers.
    """

    def __init__(self, num_classes, dropout=False, p=0.5, p_2d=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_2d) if dropout else nn.Identity(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_2d) if dropout else nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_2d) if dropout else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_2d),
            nn.AvgPool2d(kernel_size=6),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p) if dropout else nn.Identity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


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
max_epochs = 500
epochs_early_stop = 25
batch_size = 128
learning_rate = 1e-2
weight_decay = 1e-3

# reset seeds
seed()

train_ops_list = [ops.type_cast(np.float32)]
train_ops_list.append(ops.normalizePerChannel(training_set_stats["mean"], training_set_stats["std"]))

test_val_ops_list = [ops.type_cast(np.float32)]
test_val_ops_list.append(ops.normalizePerChannel(training_set_stats["mean"], training_set_stats["std"]))
test_val_ops_list.append(ops.hwc2chw())
test_val_ops = ops.chain(test_val_ops_list)

# load training and validation batches
train_set = datasets.PetsDataset(dataset_path, Subset.TRAINING)
train_ops_list.append(ops.hwc2chw())
train_batches = batches.BatchGenerator(train_set, batch_size, True, ops.chain(train_ops_list))

val_set = datasets.PetsDataset(dataset_path, Subset.VALIDATION)
val_batches = batches.BatchGenerator(val_set, batch_size, False, test_val_ops)

# create network and wrap in CnnClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CUDA, if available
num_classes = train_set.num_classes()

net = PerrosYGatosNet(num_classes, dropout=True).to(device=device)

clf = CnnClassifier(net, input_shape=input_shape, num_classes=num_classes, lr=learning_rate,
                    wd=weight_decay)

# start training procedure
train()

# load test batches
test_set = datasets.PetsDataset(dataset_path, Subset.TEST)
test_batches = batches.BatchGenerator(test_set, batch_size, False, test_val_ops)

# start testing procedure
test()
