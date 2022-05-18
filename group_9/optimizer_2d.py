import time
from collections import namedtuple

import cv2
import numpy as np
import torch
from torch import Tensor

# A 2D vector. Used in Fn as an evaluation point.
Vec2 = namedtuple('Vec2', ['x1', 'x2'])


class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimizers
    '''

    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output


def load_image(fpath: str) -> np.ndarray:
    '''
    Loads a 2D function from a PNG file and normalizes it to the interval [0, 1]
    Raises FileNotFoundError if the file does not exist.
    '''
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"{fpath} could not be found")
    return img / np.iinfo(img.dtype).max


class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fn: np.ndarray, eps: float):
        '''
        Ctor that assigns function data fn and step size eps for numerical differentiation
        '''

        self.fn = fn
        self.eps = eps

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization of the function as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        return cv2.applyColorMap(np.uint8(self.fn * 255), cv2.COLORMAP_BONE)

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # Alternatively, you can implement some form of interpolation (for example bilinear)

        # round and map to integers
        x1 = int(round(loc.x1, 0))
        x2 = int(round(loc.x2, 0))

        # perform out-of-bounds check after rounding
        if x1 < 0 or x1 >= self.fn.shape[0] or x2 < 0 or x2 >= self.fn.shape[1]:
            raise ValueError(
                f"Index out of bounds -- x1 should be between 0 and {self.fn.shape[0] - 1}, x2 should be between 0 "
                f"and {self.fn.shape[1] - 1} but got: {x1=}, {x2=}")

        return self.fn[x1, x2]

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        if self.eps <= 0:
            raise ValueError("Argument 'eps has to be > 0")

        # out-of-bounds check omitted (ValueError will be propagated by __call__)

        x1 = loc.x1
        x2 = loc.x2

        # calculate numerical gradient using the formula f'(x) = f(x+eps)-f(x-eps)/(2*eps)
        # partial derivative in direction x1
        f_x1_plus_eps = self.__call__(Vec2(x1 + self.eps, x2))
        f_x1_minus_eps = self.__call__(Vec2(x1 - self.eps, x2))
        df_x1 = (f_x1_plus_eps - f_x1_minus_eps) / (2 * self.eps)

        # partial derivative in direction x2
        f_x2_plus_eps = self.__call__(Vec2(x1, x2 + self.eps))
        f_x2_minus_eps = self.__call__(Vec2(x1, x2 - self.eps))
        df_x2 = (f_x2_plus_eps - f_x2_minus_eps) / (2 * self.eps)

        return Vec2(df_x1, df_x2)


# A point according to the OpenCV format (column, row)
OpenCVPoint = namedtuple('OpenCVPoint', ['x2', 'x1'])


def locToOpenCVPoint(t: Tensor) -> OpenCVPoint:
    """
    Converts a tensor containing a location in the format (row,column) to an OpenCVPoint (column,row)
    """
    return OpenCVPoint(round(t[1].item()), round(t[0].item()))


if __name__ == '__main__':
    # Parse args
    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--max_epochs', type=int, default=3000,
                        help='Maximum number of epochs the optimizer should run')
    parser.add_argument('--epochs_early_stop', type=int, default=50,
                        help='Number of epochs that need to elapse to stop based on the stopping criterion')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float, default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    parser.add_argument('--optimizer', type=str, default="SGD", choices=("SGD", "Adam"), help='Desired optimizer')
    parser.add_argument('--wait', type=float, default=0,
                        help='Number of seconds the canvas should be shown after the optimization is finished ('
                             'e.g., for taking screenshots)')
    parser.add_argument('--line_color', type=str, default="red", choices=("red", "orange", "green"))
    args = parser.parse_args()

    # Init
    image_fn = load_image(args.fpath)
    fn = Fn(image_fn, args.eps)
    vis = fn.visualize()

    # PyTorch uses tensors which are very similar to numpy arrays but hold additional values such as gradients
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)

    # Init chosen optimizer
    if args.optimizer == "SGD":
        print("Chosen optimizer: " + args.optimizer)
        optimizer = torch.optim.SGD([loc], lr=args.learning_rate, momentum=args.beta, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        print("Chosen optimizer: " + args.optimizer)
        # do not set beta but use default beta values
        optimizer = torch.optim.Adam([loc], lr=args.learning_rate)
    else:
        assert False  # should never happen due to 'choices' in argument parser

    # Init color for line on canvas
    if args.line_color == "red":
        color = (0, 0, 255)
    elif args.line_color == "orange":
        color = (0, 83, 255)
    elif args.line_color == "green":
        color = (0, 255, 0)
    else:
        assert False  # should never happen due to 'choices' in argument parser

    # get global minima
    global_minima = np.argwhere(image_fn == np.min(image_fn))
    print(f"Positions of global minima (x1, x2):\n{global_minima}")

    # Find a minimum in fn using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use optimizers
    epoch = 0
    curr_min_loss = float("inf")
    num_epochs_no_improvement = 0
    stop_position = None
    while True:
        epoch += 1

        optimizer.zero_grad()

        start_point = locToOpenCVPoint(loc)

        try:
            # This returns the value of the function fn at location loc.
            # Since we are trying to find a minimum of the function this acts as a loss value.
            loss = AutogradFn.apply(fn, loc)
            loss.backward()
        except ValueError as e:
            print(
                f"Epoch {epoch}: Function is undefined at this position (stop position {start_point.x1}, "
                f"{start_point.x2} (x1,x2))")
            stop_position = start_point
            time.sleep(args.wait)
            break

        optimizer.step()

        end_point = locToOpenCVPoint(loc)

        # Visualize each iteration by drawing on vis
        cv2.line(vis, start_point, end_point, color, thickness=2)
        cv2.imshow('Progress', vis)
        cv2.waitKey(1)  # 20 fps, tune according to your liking

        # stop when no improvement in loss (e.g., decrease in loss) for args.epochs_early_stop
        if loss.item() < curr_min_loss:
            curr_min_loss = loss.item()
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
            if num_epochs_no_improvement == args.epochs_early_stop:
                print(
                    f"Epoch {epoch}: Stopping criterion fulfilled. No improvement in loss for "
                    f"{args.epochs_early_stop} consecutive epochs (stop position {end_point.x1}, {end_point.x2} (x1,"
                    f"x2))")
                stop_position = end_point
                time.sleep(args.wait)
                break

        # stop if maximum number of epochs is reached
        if epoch == args.max_epochs:
            print(
                f"Epoch {epoch}: Reached maximum number of epochs (stop position {end_point.x1}, {end_point.x2} (x1,"
                f"x2))")
            stop_position = end_point
            time.sleep(args.wait)
            break

    # print message to console if a global minimum was reached
    if np.any(np.all(global_minima == np.array((stop_position.x1, stop_position.x2)), axis=1)):
        print("Global minimum reached")
