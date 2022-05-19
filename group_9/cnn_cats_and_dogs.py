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


class BaselineCNN(nn.Module):
    """
    CNN model from the PyTorch Tutorial with adapted linear layers.
    Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, Accessed: 2022-05-18
    """

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PerrosYGatosNet(nn.Module):
    """
    Our own CNN architecture -- Perros-y-Gatos-Net -- a CNN for classifying cats and dogs images.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=6),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train():
    val_acc_best = Accuracy()
    for e in range(args.max_epochs):
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

        print(val_acc_curr)

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


if __name__ == '__main__':
    # parse arguments (needed/re-used in cnn_cats_and_dogs_pt3.py)
    parser = argparse.ArgumentParser(
        description='Train a CNN for cats and dogs images using Stochastic Gradient Descent (SGD) with Nesterov '
                    'Momentum')
    parser.add_argument('--dataset_path', type=str, default=os.path.join(os.pardir, "cifar-10-batches-py"))
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for SGD')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for SGD')
    parser.add_argument('--net', type=str, default="PerrosYGatosNet", choices=("BaselineCNN", "PerrosYGatosNet"), help='CNN architecture/network that should be used')
    parser.add_argument('--per_channel_norm', action='store_true', help='Use per-channel normalization')
    args = parser.parse_args()

    print("====================\nSETUP\n====================")
    print("".join([f"{k}:{v}\n" for k, v in vars(args).items()]), end="")  # TODO: format output
    print("====================")

    model_filepath = os.path.join(os.curdir, f"ex2_pt2_{args.net}_model.pt")

    # reset seeds
    seed()

    ops_chain = [ops.type_cast(np.float32)]
    if args.per_channel_norm:
        ops_chain.append(ops.normalizePerChannel(training_set_stats["mean"], training_set_stats["std"]))
    else:
        ops_chain.append(ops.add(-127.5))
        ops_chain.append(ops.mul(1 / 127.5))
    ops_chain.append(ops.hwc2chw())
    op = ops.chain(ops_chain)

    # load training and validation batches
    train_set = datasets.PetsDataset(args.dataset_path, Subset.TRAINING)
    train_batches = batches.BatchGenerator(train_set, args.batch_size, True, op)

    val_set = datasets.PetsDataset(args.dataset_path, Subset.VALIDATION)
    val_batches = batches.BatchGenerator(val_set, args.batch_size, False, op)

    # create network and wrap in CnnClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CUDA, if available
    num_classes = train_set.num_classes()

    if args.net == "BaselineCNN":
        net = BaselineCNN(num_classes).to(device=device)
    elif args.net == "PerrosYGatosNet":
        net = PerrosYGatosNet(num_classes).to(device=device)
    else:
        assert False  # should never happen due to 'choices' in argument parser

    clf = CnnClassifier(net, input_shape=input_shape, num_classes=num_classes, lr=args.learning_rate,
                        wd=args.weight_decay)

    # start training procedure
    train()

    # load test batches
    test_set = datasets.PetsDataset(args.dataset_path, Subset.TEST)
    test_batches = batches.BatchGenerator(test_set, args.batch_size, False, op)

    # start testing procedure
    test()
