import os
import random

import dlvc.batches as batches
import dlvc.datasets.pets as datasets
from dlvc.dataset import Subset
import dlvc.ops as ops
from dlvc.test import Accuracy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the network architecture of the linear classifier
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# Ensure reproducibility
# Source: https://pytorch.org/docs/stable/notes/randomness.html (Accessed: 2022-04-11)
SEED = 29
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create BatchGenerator for training, validation and test datasets
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
])

DATASET_PATH = os.path.join(os.pardir,
                            "cifar-10-batches-py")  # assumes that CIFAR-10 dataset is placed in parent folder
MODEL_FILENAME = "model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use CUDA, if available

train = datasets.PetsDataset(DATASET_PATH, Subset.TRAINING)
train_batches = batches.BatchGenerator(train, len(train), True, op)

val = datasets.PetsDataset(DATASET_PATH, Subset.VALIDATION)
val_batches = batches.BatchGenerator(val, len(val), False, op)

test = datasets.PetsDataset(DATASET_PATH, Subset.TEST)
test_batches = batches.BatchGenerator(test, len(test), False, op)

# Create linear classifier, loss function and optimizer
model = LinearClassifier(3072, train.num_classes()).to(device=DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the 
training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your 
findings in the report.
'''
num_epochs = 150
val_acc_best = Accuracy()

# train and validate
for e in range(num_epochs):
    model.train()
    avg_train_loss = 0.0
    for train_batch in train_batches:
        optimizer.zero_grad()

        data = torch.tensor(train_batch.data).to(device=DEVICE)
        labels = torch.tensor(train_batch.label).to(device=DEVICE)

        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item() * data.size(0)

    avg_train_loss /= len(train_batches.dataset)

    model.eval()
    with torch.no_grad():
        val_acc_curr = Accuracy()
        for val_batch in val_batches:
            data = torch.tensor(val_batch.data).to(device=DEVICE)
            output = model(data)
            val_acc_curr.update(output.cpu().detach().numpy(), val_batch.label)

        if val_acc_curr > val_acc_best:
            val_acc_best = val_acc_curr
            torch.save(model.state_dict(), os.path.join(os.curdir, MODEL_FILENAME))

    print(f"epoch {e + 1}\ntrain loss: {avg_train_loss:.3f}\nval {val_acc_curr}")

print(f"--------------------\nval acc (best): {val_acc_best.accuracy():.3f}")

# Test with model that obtained the highest accuracy on the validation set
model.load_state_dict(torch.load(os.path.join(os.curdir, MODEL_FILENAME)))
model.eval()
test_acc = Accuracy()

with torch.no_grad():
    for test_batch in test_batches:
        data = torch.tensor(test_batch.data).to(device=DEVICE)
        output = model(data)
        test_acc.update(output.cpu().detach().numpy(), test_batch.label)

print(f"test {test_acc}")
