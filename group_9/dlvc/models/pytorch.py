import numpy as np
import torch.nn as nn
import torch.optim

from ..model import Model


class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''
        # Inside the train() and predict() functions you will need to know whether the network itself
        # runs on the CPU or on a GPU, and in the latter case transfer input/output tensors via cuda() and cpu().
        # To termine this, check the type of (one of the) parameters, which can be obtained via parameters() (there
        # is an is_cuda flag).
        # You will want to initialize the optimizer and loss function here.
        # Note that PyTorch's cross-entropy loss includes normalization so no softmax is required

        pass

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        # TODO implement

        pass

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # TODO implement

        pass

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Make sure to set the network to train() mode
        # See above comments on CPU/GPU

        # check for type errors
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Argument 'data' has invalid type. Actual: {type(data)}. Expected {np.ndarray}.")

        if data.dtype != np.float32:
            raise TypeError(f"Argument 'data' has invalid dtype. Actual: {data.dtype}. Expected: {np.float32}.")

        if not isinstance(labels, np.ndarray):
            raise TypeError(f"Argument 'labels' has invalid type. Actual: {type(labels)}. Expected {np.ndarray}.")

        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"Argument 'labels' has invalid dtype. Actual: {labels.dtype}. Expected: integral type dtype.")

        # check for value errors
        if len(data.shape) != 4 or data.shape[1:] != self._input_shape[1:]:
            raise ValueError(
                f"Argument 'data' has invalid shape. Actual: {data.shape}. Expected: {self._input_shape} (N, C, H, "
                f"W) where N (indicated by 0) is the variable batch size.")

        if len(labels.shape) != 1:
            raise ValueError(f"Argument 'labels' has invalid shape. Actual: {labels.shape}. Expected: (m,).")

        labels_unique = np.unique(labels)
        if labels_unique[0] != 0 or labels_unique[-1] != (self._num_classes - 1):
            raise ValueError(
                f"Argument 'labels' has wrong value range. Actual: {labels_unique[0]}-{labels_unique[-1]}. Expected: "
                f"0-{self._num_classes - 1}")

        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"First dimension of argument 'data' and 'labels' does not match. Actual: 1st dim data "
                f"{data.shape[0]}, 1st dim labels {labels.shape[0]}. Expected: equal numbers.")

        try:
            self._net.train()
            self._optimizer.zero_grad()

            data = torch.from_numpy(data).to(device=self._device)
            labels = torch.from_numpy(labels).to(device=self._device)

            output = self._net(data)
            loss = self._criterion(output, labels)
            loss.backward()
            self._optimizer.step()

            return loss.item()
        except Exception as e:
            raise RuntimeError("Error thrown by PyTorch in train():" + str(e))

    @torch.no_grad()
    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Pass the network's predictions through a nn.Softmax layer to obtain softmax class scores
        # Make sure to set the network to eval() mode
        # See above comments on CPU/GPU

        # check for type errors
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Argument 'data' has invalid type. Actual: {type(data)}. Expected: {np.ndarray}")

        if data.dtype != np.float32:
            raise TypeError(f"Argument 'data' has invalid dtype. Actual: {data.dtype}. Expected: {np.float32}")

        # check for value errors
        if len(data.shape) != 4 or data.shape[1:] != self._input_shape[1:]:
            raise ValueError(
                f"Argument 'data' has invalid shape. Actual: {data.shape}. Expected: {self._input_shape} (N, C, H, "
                f"W) where N (indicated by 0) is the variable batch size.")

        try:
            self._net.eval()
            data = torch.from_numpy(data).to(device=self._device)
            pred = self._net(data)
            output = self._softmax(pred)

            return output.detach().cpu().numpy()
        except Exception as e:
            raise RuntimeError("Error thrown by PyTorch in predict():" + str(e))
