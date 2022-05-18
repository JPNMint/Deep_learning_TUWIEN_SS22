import argparse
import os
import random
from statistics import mean, stdev

import dlvc.batches as batches
import dlvc.datasets.pets as datasets
from dlvc.dataset import Subset
import dlvc.ops as ops
from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BaselineCNN(nn.Module):
    """
    Baseline CNN model from the PyTorch Tutorial.
    Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, Accessed: 2022-05-18
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    val_acc_best = Accuracy()
    for e in range(args.max_epochs):
        print(f"Epoch {e + 1}")

        train_loss = []
        for train_batch in train_batches:
            data = train_batch.data
            labels = train_batch.label
            train_loss.append(clf.train(data, labels))

        # Note: Calculating the mean and standard deviation of the loss this way may introduce a slight bias if the
        # number of samples in the dataset is not a multiple of the batch size
        print(f"train loss: {mean(train_loss):.3f} ± {stdev(train_loss):.3f}")

        val_acc_curr = Accuracy()
        for val_batch in val_batches:
            data = val_batch.data
            label = val_batch.label
            output = clf.predict(data)
            val_acc_curr.update(output, label)

        if val_acc_curr > val_acc_best:
            val_acc_best = val_acc_curr
            torch.save(net.state_dict(), model_filepath)

        print(val_acc_curr)

    print(f"--------------------\nval acc (best): {val_acc_best.accuracy():.3f}")


def test():
    net.load_state_dict(torch.load(os.path.join(os.curdir, model_filepath)))
    net.eval()
    test_acc = Accuracy()

    with torch.no_grad():
        for test_batch in test_batches:
            data = test_batch.data
            label = test_batch.label
            output = clf.predict(data)
            test_acc.update(output, label)

    print(f"test acc {test_acc.accuracy()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a CNN for cats and dogs images using Stochastic Gradient Descent (SGD) with Nesterov '
                    'Momentum')
    parser.add_argument('--dataset_path', type=str, default=os.path.join(os.pardir, "cifar-10-batches-py"))
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for SGD')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for SGD')
    args = parser.parse_args()

    print("====================\nSETUP\n====================")
    print("".join([f"{k}:{v}\n" for k, v in vars(args).items()]), end="")  # TODO: format output
    print("====================")

    model_filepath = os.path.join(os.curdir, "saved_models",
                                  f"model_batch-size-{args.batch_size}_lr-{args.learning_rate}_wd-{args.weight_decay}.pt")

    # reset seeds
    seed()

    op = ops.chain([
        ops.type_cast(np.float32),
        ops.add(-127.5),
        ops.mul(1 / 127.5),
        ops.hwc2chw()
    ])

    # load training and validation batches
    train_set = datasets.PetsDataset(args.dataset_path, Subset.TRAINING)
    train_batches = batches.BatchGenerator(train_set, args.batch_size, True, op)

    val_set = datasets.PetsDataset(args.dataset_path, Subset.VALIDATION)
    val_batches = batches.BatchGenerator(val_set, args.batch_size, False, op)

    # create network and wrap in CnnClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CUDA, if available
    num_classes = train_set.num_classes()
    net = BaselineCNN(num_classes).to(device=device)
    clf = CnnClassifier(net, input_shape=(0, 3, 32, 32), num_classes=num_classes, lr=args.learning_rate,
                        wd=args.weight_decay)

    # start training procedure
    train()

    # load test batches
    test_set = datasets.PetsDataset(args.dataset_path, Subset.TEST)
    test_batches = batches.BatchGenerator(test_set, args.batch_size, False, op)

    # start testing procedure
    test()
