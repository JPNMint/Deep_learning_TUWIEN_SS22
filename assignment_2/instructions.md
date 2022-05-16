# Deep Learning for Visual Computing - Assignment 2

The second assignment covers iterative optimization and parametric (deep) models for image classification.

## Part 1

This part is about experimenting with different flavors of gradient descent and optimizers.

Download the data from [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/tree/main/assignments/assignment_2). Your task is to implement `optimizer_2d.py`. We will use various optimization methods implemented in PyTorch to find the minimum in a 2D function given as an image. In this scenario, the optimized weights are the coordinates at which the function is evaluated, and the loss is the function value at those coordinates.

See the code comments for instructions. The `fn/` folder contains sampled 2D functions for use with that script. For bonus points you can add and test your own functions (something interesting with a few local minima). For this you don't necessarily have to use `load_image`, you can also write a different function that generates a 2D array of values.

The goal of this part is for you to better understand the optimizers provided by PyTorch by playing around with them. Try different types (SGD, AdamW etc.), parameters, starting points, and functions. How many steps do different optimizers take to terminate? Is the global minimum reached? What happens when weight decay is set to a non-zero value and why? This nicely highlights the function and limitations of gradient descent, which we've already covered in the lecture.

## Part 2

Time for some Deep Learning. We already implemented most of the required functionality during Assignment 1. Make sure to fix any mistakes mentioned in the feedback you received for your submission. With the exception of `linear_cats_and_dogs.py` all files will be reused in this assignment. The main thing that is missing is a subtype of `Model` that wraps a PyTorch CNN classifier. Implement this type, which is defined inside `dlvc/models/pytorch.py` and named `CnnClassifier`. Details are stated in the code comments. The PyTorch documentation of `nn.Module`, which is the base class of PyTorch models, is available [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

PyTorch (and other libraries) expects the channel dimension of a single sample to be the first one, rows the second one, and columns the third one (`CHW` for short). However in our case they are `HWC`. To address this, implement the `hwc2chw()` function in `ops.py` (make sure to download the updated reference code).

Once this is in place, create a script named `cnn_cats_and_dogs.py`. This file will be very similar to the version for the linear classifier (`linear_cats_and_dogs.py`) developed for Assignment 1 so you might want to use that one as a reference. This file should implement the following in the given order:

1. Load the training, validation and test subsets of the `PetsDataset`.
2. Initialize `BatchGenerator`s for both with batch sizes of 128 or so (feel free to experiment) and the input transformations required for the CNN. This should include input normalization. A basic option is `ops.add(-127.5), ops.mul(1/127.5)` but for bonus points you can also experiment with more sophisticated alternatives such as per-channel normalization using statistics from the training set (if so create corresponding operations in `ops.py` and document your findings in the report).
3. Define a PyTorch CNN with an architecture suitable for cat/dog classification. To do so create a subtype of `nn.Module` and overwrite the `__init__()` and `forward()` methods (do this inside `cnn_cats_and_dogs.py`). If you have access to an Nvidia GPU transfer the model using the `.cuda()` method of the CNN object.
4. Wrap the CNN object `net` in a `CnnClassifier`, `clf = CnnClassifier(net, ...)`.
5. Inside a `for epoch in range(100):` loop (i.e. train for 100 epochs which is sufficient for now), train `clf` on the training set and store the losses returned by `clf.train()` in a list. Then convert this list to a numpy array and print the mean and standard deviation in the format `mean ± std`. Then print the accuracy on the validation set using the `Accuracy` class developed in Assignment 1. While training, keep track of the best performing model with respect to validation accuracy and save it. At the end of the run compute the accuracy on the test subset and print it out as well. 

The console output should thus be similar to the following (ignoring the values):
```python
epoch 1
train loss: 0.689 ± 0.006
val acc: 0.561
epoch 2
train loss: 0.681 ± 0.008
val acc: 0.578
epoch 3
train loss: 0.673 ± 0.009
val acc: 0.585
epoch 4
train loss: 0.665 ± 0.013
val acc: 0.594
epoch 5
train loss: 0.658 ± 0.014
val acc: 0.606
--------------------
val acc (best): 0.606
test acc: 0.612
...
```

The goal of this part is for you to get familiar with PyTorch and to be able to try out different architectures and layer combinations. The pets dataset is ideal for this purpose because it is small. Experiment with the model by editing the code manually rather than automatically via hyperparameter optimization. What you will find is that the training loss will approach 0 even with simple architectures (demonstrating how powerful CNNs are and how well SGD works with them) while the validation accuracy will likely not exceed 75%. The latter is due to the small dataset size, resulting in overfitting. We will address this in the next part.

## Part 3

Address the overfitting issue of part 2 using a combination of the techniques we covered in the lecture, namely data augmentation, regularization, early stopping, and transfer learning. To get all points you at least have to utilize the first three:

* Data augmentation: implement (and use) at least random crops and left/right mirroring. The corresponding operations are defined in `ops.py`. You can earn bonus points if you find and implement other operations that increase validation accuracy.
* Regularization: use at least weight decay but you can also experiment with dropout (in addition to or instead of weight decay).
* For early stopping you can simply [save](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) the current model to disk to something like `best_model.pt` if it has the highest validation accuracy seen so far (and overwrite that file in the process of training).

In the report you should discuss how the individual techniques affected your training and validation set accuracy. Don't just compare part 2 and part 3 results, also compare at least a few combinations and settings like only regularization but no data augmentation vs. both, different regularization strengths and so on. **This may take some time, so don't delay working on the assignment until shortly before the deadline.** Try a couple combinations (around 5).

Submit the configuration that leads to the best validation accuracy as a file `cnn_cats_and_dogs_pt3.py`. This file should be based on `cnn_cats_and_dogs.py` from part 2, i.e. it should train the CNN and output the training and validation accuracy in every epoch. You must also submit the corresponding model you obtained (`best_model.pt`).

For bonus points you can additionally experiment with transfer learning (see e.g. [here](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)). For easy integration into the existing code you will want to replace the classification layers, freeze all other parameters (see "feature extraction" and the `requires_grad` attribute in [this](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#set-model-parameters-requires-grad-attribute) section) and then simply wrap the net inside `CnnClassifier`. This way you can reuse all code as you did in part 2. You can also try the "finetuning" option described in the previous link but note that this option is much more computationally expensive.

## Report

Write a short report (3 to 4 pages) that answers the following questions:

* How does gradient descent work? What are your findings on the example data (part 1)?
* Which network architecture did you choose for part 2 and why? Did you have any problems reaching a low training error?
* What are the goals of data augmentation, regularization, and early stopping? How exactly did you use these techniques (hyperparameters, combinations) and what were your results (train and val performance)? List all experiments and results, even if they did not work well, and discuss them.
* If you utilized transfer learning, explain what you did and your results.

## Submission

Submit your assignment until **May 20 at 11pm**. To do so, create a zip archive including the report, the complete `dlvc` folder with your implementation and all scripts. More precisely, after extracting the archive we should obtain the following:

    group_x/
        report.pdf
        optimizer_2d.py
        cnn_cats_and_dogs.py
        cnn_cats_and_dogs_pt3.py
        dlvc/
            ...

Submit the zip archive in TUWEL. Make sure you've read the general assignment information [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/blob/main/assignments/general.md) before your final submission.

## Server Usage

You may find that training is slow on your computer unless you have an Nvidia GPU with CUDA support. If so, copy the code into your home directory on the DLVC server and run it there. The login credentials will be sent out on April 28th - check your spam folder if you didn't. For details on how to run your scripts see [here](https://smithers.cvl.tuwien.ac.at/jstrohmayer/dlvc_ss22/-/blob/main/assignments/DLVC2022Guide.pdf). For technical problems regarding our server please contact [email](mailto:dlvc-trouble@cvl.tuwien.ac.at).

We expect queues will fill up close to assignment deadlines. In this case, you might have to wait a long time before your script even starts. In order to minimize wait times, please do the following:

* Write and test your code locally on your system. If you have a decent Nvidia GPU, please train locally and don't use the servers. If you don't have such a GPU, perform training for a few epochs on the CPU to ensure that your code works. If this is the case, upload your code to our server and do a full training run there. To facilitate this process, have a variable or a runtime argument in your script that controls whether CUDA should be used. Disable this locally and enable it on the server.
* Don't schedule multiple training runs in a single job, and don't submit multiple long jobs. Be fair.
* If you want to train on the server, do so as early as possible. If everyone starts two days before the deadline, there will be long queues and your job might not finish soon enough.
