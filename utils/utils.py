import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from numba import jit
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt

from numba import njit, prange, cuda, float32, set_num_threads, get_num_threads
from numpy.testing import assert_almost_equal, assert_allclose
import logging

parallel_range = True
use_fastmath = True


def configure_logger(debug_mode=True, log_results_file = 'experiment_results.log',
                     log_debug_file = 'debug.log'):
    """Make a customized logger for the experiments.

    Configure logger:
    
    logger = configure_logger()

    Log a result to the log_results_file:
    logger.info(f"Experiment 1: Accuracy = {accuracy}")

    Log debug information to the log_debug_file:
    logger.debug("Experiment 1: Debug information here.")
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    print("Setting of logger.")
    print(f"Logging experiment results to: {log_results_file}")
    print(f"Logging debug info to: {log_debug_file}")
    info_file_handler = logging.FileHandler(log_results_file)
    debug_file_handler = logging.FileHandler(log_debug_file)

    info_file_handler.setLevel(logging.INFO)
    debug_file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    info_file_handler.setFormatter(formatter)
    debug_file_handler.setFormatter(formatter)

    logger.addHandler(info_file_handler)

    if debug_mode:
        logger.addHandler(debug_file_handler)

    return logger



# from: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class ValueAccumulator:
    def __init__(self):
        self.count = 0
        self.total = 0

    def accumulate(self, value, count=1):
        self.total += value
        self.count += count

    def get_total(self):
        return self.total

    def get_mean(self):
        if self.count == 0:
            raise RuntimeError("No values yet!")
        return self.total / self.count

    def get_count(self):
        return self.count

    def get_problem_count(self):
        return self.inf_count + self.nan_count

    def reset(self):
        self.count = 0
        self.total = 0


###################################################################
# Plotting functions


def plot_image_matrix(x, file_name="temp.png", title="todo: add the title", 
                      origin="upper", vmin=None, vmax=None,
                      xlabel = None,
                      ylabel = None,
                      include_colorbar = True
):
    """Wrapper around matplotlib imshow function with colorbar.

    Plot the matrix 'x' and save it to a file.
    x should have shape: (h w c) or (h w)

    The "hot" colormap is used by default.

    """
    if isinstance(x, torch.Tensor):
        x = x.to("cpu")
        x = x.detach().numpy()
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    fig1, ax1 = plt.subplots()
    #im = ax1.imshow(
    #    x, origin=origin, cmap=plt.cm.hot, aspect="auto", interpolation="nearest"
    #)
    im = ax1.imshow(
        x, origin=origin, cmap=plt.cm.hot, aspect="auto", 
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )
    if include_colorbar:
        plt.colorbar(im)
    ax1.set_title(title)
    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    if ylabel is not None:
        ax1.set_ylabel(ylabel)
    fig1.savefig(file_name, dpi=300)
    plt.close(fig1)


def torch_to_numpy_cpu(x):
    """Convert torch tensor on CPU or GPU to Numpy array on CPU.

    If it is already a Numpy array, do nothing.

    """
    if isinstance(x, torch.Tensor):
        x = x.to("cpu")
        x = x.detach().numpy()
    return x


def plot_wrapper(
    x, y=None, file_name="temp.png", title="todo", x_label="My x-label", y_label="My y-label"
):
    """Plot x versus y.

    Plot x versus y and save the plot to a file instead of displaying it in a window.
    If y is None, only plot x.
    """
    if len(x) == 0:
        return
    x = torch_to_numpy_cpu(x)
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    if y is None:
        ax1.plot(x)
    else:
        ax1.plot(x, y)
    ax1.set_title(title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    fig1.savefig(file_name, dpi=150)
    plt.close(fig1)


def plot_rows_as_images(
    images, file_name="temp.png", img_height=28, img_width=28, img_channels=1, 
    plot_image_count=None,
    normalize=False
):
    """Plot each row in the supplied matrix as an image in a grid plot.

    Each row in 'images' is a vector that represents one image.

    Each row has dimension `row_dim` which can contain a flattened greyscale or
    color image.
    If the image is grescale, the row will be reshaped to (h, w) and plotted.
    If the image is color, the row will be reshaped to (c, h, w) where c=3
    and plotted.
    
    Args:
        images (ndarray or torch tensor.): tensor of shape (batch_size, row_dim)
            where row_dim = height*width or 3*height*width.

    """
    if isinstance(images, torch.Tensor):
        images = images.to("cpu")
        images = images.detach().numpy()
    if plot_image_count is None:
        plot_image_count = images.shape[0]

    assert plot_image_count <= images.shape[0]
    if False:
        rows = int(np.sqrt(plot_image_count))
        cols = rows
    if True:
        rows = int(np.sqrt(plot_image_count))
        if rows*rows < plot_image_count:
            cols = int(np.ceil(plot_image_count / rows))
        else:
            cols = rows
    #assert images.min() >= 0.0, "Found negative values in iamges!"
    if images.min() < 0:
        print("Warning: found negative values in the image plot data!")
        images[images < 0] = 0
    max_val = images.max()
    images = images / max_val

    for n in range(plot_image_count):
        if img_channels == 1:
            cur_image = images[n].reshape(img_height, img_width)
            local_max_val = cur_image.max()
            if normalize and local_max_val > 0:
                cur_image = cur_image / local_max_val
        elif img_channels == 3:
            cur_row = images[n]
            cur_image = rearrange(cur_row, "(c h w) -> h w c", h=img_height, w=img_width, c=3)
            local_max_val = cur_image.max()
            if normalize and local_max_val > 0:
                cur_image = cur_image / local_max_val
        else:
            raise ValueError("oops, bad number of channels")
        plt.subplot(rows, cols, n + 1)
        plt.axis("off")
        #plt.imshow(cur_image, cmap="hot", interpolation="nearest")
        assert cur_image.max() <= 1.0
        assert local_max_val <= 1.0
        if normalize and local_max_val > 0:
            plt.imshow(cur_image, cmap="hot", interpolation="nearest", vmax=local_max_val)
        else:
            #plt.imshow(cur_image, cmap="hot", interpolation="nearest", vmax=max_val)
            plt.imshow(cur_image, cmap="hot", interpolation="nearest", vmax=1.0)
            #plt.imshow(cur_image, cmap="hot", interpolation="nearest")
    # plt.show()
    plt.savefig(file_name, dpi=200)
    plt.close()

###################################################################
# FISTA optimization-related functions

# Only for debug. To verify that the power_method() function is accurate.
def exact_largest_eigenvalue(X):
    eigs = torch.linalg.eig(X).eigenvalues
    eigs_mags = torch.abs(eigs)
    max_eigv = eigs_mags.max(dim=-1)
    return max_eigv.values

# (with help from ChatGPT4)
def power_method(A, num_iterations=40, tolerance=1e-5, debug = False):
    """
    The Power Method is an iterative algorithm used for estimating the 
    largest (in magnitude) eigenvalue of a matrix. 

    This function takes as input a square matrix A, a number of iterations 
    to run (default 1000), and a tolerance for checking convergence (default 1e-5). 
    It returns an estimate of the largest eigenvalue of A and the corresponding eigenvector.

    """
    num_rows, num_cols = A.shape
    assert num_rows == num_cols, "Matrix must be square."
    v = torch.rand(num_cols, device=A.device)
    
    for n in range(num_iterations):
        v_next = torch.mv(A, v)
        v_next /= v_next.norm()
        if torch.dist(v, v_next) < tolerance:
            v = v_next
            if debug:
                print(f"Power method iterations: {n}")
            break

        v = v_next

    # Estimate of the largest eigenvalue is given by the Rayleigh quotient
    lambda_estimate = v.dot(A.mv(v)) / v.dot(v)

    return lambda_estimate, v


@torch.no_grad()
def fista_compute_lr_from_weights(W, use_power_method=True):
    """Compute the FISTA learning rate from the weights matrix W.
    
    Consider the following matrix factorization:

    X approx= W * H

    Compute the learning rate lr_H for the NMF right update rule:

    When using FISTA to compute the right update for H, we need to compute
    it as lr_H = 1/L where L is the largest eigenvalue of W^T*W.

    In this case, we can compute the learning rate as:

    lr_H = fista_compute_lr_from_weights(W)

    This function uses the power method to compute the approximate largest eigenvalue L.
    The learning rate is then returned as lr_H = 1/L.

    The computed learning rate can also be used with the default SGD NMF right update
    rule.

    Compute the learning rate lr_W for the NMF left update rule:

    Note from symmetry of the factorization that we can take the transpose of
    both sides:

    X^T approx= H^T * W^T

    This implies that we can then compute the learning rate lr_W for the 
    NMF left update rule as follows:

    lr_W = fista_compute_lr_from_weights(H_tran)

    where H_tran is the transpose of H.

    Note: This can also be used for the general case without the non-negativity
    constraint.

    Args:
        W: The weights W in the NMF factorization.
        use_power_method (bool): If True, use the power method. Otherwise,
            compute the exact eigenvalue, which will likely be much slower.

    """
    W_tran_W = W.t() @ W
    if use_power_method:
        L, _ = power_method(W_tran_W)
    else:
        L = exact_largest_eigenvalue(W_tran_W)
    #print(f"Largest eigenvalue L: {L}")
    lr = 1.0 / L
    return lr.item()



def fista_momentum(cur_H, prev_H, cur_m):
    """The FISTA momentum update step.
    """
    next_m = (1 + torch.sqrt(1+4*(cur_m**2)))/2.0
    ratio_m = (cur_m - 1) / next_m
    next_H = cur_H + ratio_m * (cur_H - prev_H)
    return next_H, next_m


def fista_right_update(X, W, H_tran_init=None, tolerance = 0.001, max_iterations=25, min_val = 1e-5, 
                           return_iter_count = False, shrinkage_sparsity = 0,
                           force_nonnegative=True,
                           apply_normalization_scaling = False,
                           debug=False,
                           logger=None):
    """Matrix factorization right update rule using FISTA.
    
    This runs several iterations of FISTA to compute the update for H in

    X approx= W * H

    and returns the transpose of H.

    This function initializes H to zeros and then performs several iterations of FISTA
    until approximate convergence.

    Note: `The tolerance` argument is used to bail out of the iteration once the solution
        has approximately converged. Typically values are 1e-2 to 1e-3 or so.

    Args:
        X: Data matrix of factorization.
        W: Weights matrix of factorization.
        H_tran_init (torch.tensor or None): Initial values for the inferred activation encoding matrix. 
            This is the transpose of H in the factorization.
        tolerance (float or None): The iterations will end once the relative L2 norm difference 
            between two consecutive
            H estimate drops below the specified tolerance. To disable tolerance and
            always run `max_iterations` iterations, set to `None`.
        max_iterations (int): Maximum number of iterations to compute. This value will always be used if
            `tolerance` is `None`.
        min_val: Smallest allowable value in the returned H. Suggest 1e-5 or so.
        force_nonnegative: If True, perform NMF updates so that the returned H_tran will
            contain only non-negative values.
        apply_normalization_scaling (bool): If `True`, apply "Normalization for unrolled inference with backpropagation"
            as discussed in the paper. The basic ideas is: Do not let the inferred values in a given column of `H`
            from exceeding the maximum value of the corresponding column of `X`. 
        
    Returns:
        H_tran (torch.tensor): The updated transpose of H.

    """
    with torch.no_grad():
        lr = fista_compute_lr_from_weights(W)
        #lr = lr*0.5 # todo: is less then 1.0 better?
        #print(f"fista lr: {lr}")
    if debug:
        #print(f"lr from power method: {lr}")
        logger.debug(f"fista_right_update(): lr from power method: {lr}")
    (feature_dim, template_count) = W.size()
    (feature_dim_X, num_samples) = X.size()
    assert feature_dim == feature_dim_X
    if H_tran_init is None:
        prev_H_tran = torch.zeros((num_samples, template_count), dtype=X.dtype, device=X.device)
        next_H_tran = torch.ones((num_samples, template_count), dtype=X.dtype, device=X.device)*min_val
    else:
        prev_H_tran = H_tran_init
        next_H_tran = H_tran_init
    momentum = torch.ones(1, device=X.device)

    if max_iterations == 0:
        if return_iter_count:
            return next_H_tran, n+1
        else:
            return next_H_tran

    if apply_normalization_scaling:
        # Normalize "per NMF example"
        # This is a very simple way to keep h from exploding: In X approx= W * H, each column x_i of X is
        # an "example" and each corresponding column h_i of H is the inferred basis activations for the
        # same (i'th) example. So, we just limit h_i such that its largest value is not allowed to be larger
        # than the largest value in x_i.
        with torch.no_grad():
            # Get max value in each column of X.
            max_allowed_example_vals = X.max(dim=0)[0]

    for n in range(max_iterations):
        cur_H_tran = right_update_nmf_sgd(
                    X,
                    W,
                    next_H_tran,
                    learn_rate=lr,
                    shrinkage_sparsity=shrinkage_sparsity,
                    force_nonnegative_H=force_nonnegative,
                )

        if apply_normalization_scaling:
            # Normalize the rows of H_tran (i.e., the columns of H) to have max values that are no larger than the corresponding
            # max values of the columns of the "X" matrix.
            with torch.no_grad():
                maxh_columns = cur_H_tran.max(dim=1)[0]
                scale = torch.where(maxh_columns > max_allowed_example_vals, max_allowed_example_vals / maxh_columns, torch.tensor(1., device=cur_H_tran.device)).view(-1, 1)
            cur_H_tran = cur_H_tran * scale
        
        next_H_tran, momentum = fista_momentum(cur_H_tran, prev_H_tran, momentum)
        
        if tolerance is not None:
            tolerance_achieved = relative_norm_difference(prev_H_tran, cur_H_tran)
            if tolerance_achieved < tolerance:
                if debug:
                    #print(f"FISTA converged in {n+1} iterations.")
                    pass
                break

        #if debug:
        #    if n % 10 == 0:
        #        # print RMSE
        #        x_pred = W @ cur_H_tran.t()
        #        rmse = compute_rmse(X, x_pred)
        #        tolerance_achieved = relative_norm_difference(prev_H_tran, cur_H_tran)
        #        print(f"FISTA iteration: {n} | RMSE: {rmse} | tolerance: {tolerance_achieved}")

        prev_H_tran = cur_H_tran

    if max_iterations == 0:
        cur_H_tran = next_H_tran

    #if debug:
    #    tolerance_achieved = relative_norm_difference(prev_H_tran, cur_H_tran)
    #    print(f"tolerance achieved: {tolerance_achieved}")
    if return_iter_count:
        return cur_H_tran, n+1
    else:
        return cur_H_tran


def fista_nmf_left_update(X, W, H_tran, tolerance = None, max_iterations=20, min_val = 1e-5, 
                           return_iter_count = False,
                            force_nonnegative = True, debug=False):
    """NMF left update rule using FISTA.
    
    This runs several iterations of FISTA to compute the update for W in

    X approx= W * H

    and returns the updated W.

    Note that this version requires initial matrices for both W and H. Note that
    while fista_nmf_right_update() initializes H to zeros and iterates to a solution,
    this function is different in that instead of a zero matrix for W, it uses the
    user-supplied W as the initial estimate and then iterates to a new approximate
    fixed point for W, which is returned.

    Args:
        X: Data matrix in the factorization.
        W: This initial/existing estimate for `W`, which will be refined by this function.
        H_tran: The transpose of `H`.
        tolerance (float or `None`): The iterations will end once the relative L2 norm difference 
            between two consecutive
            `W` estimate drops below the specified tolerance. To disable tolerance and
            always run `max_iterations` iterations, set to `None`.
        max_iterations (int): Maximum number of iterations to compute. This value will always be used if
            `tolerance` is `None`.
        min_val: Smallest allowable value in the returned W.
    Returns:
        The updated W.

    """
    with torch.no_grad():
        lr = fista_compute_lr_from_weights(H_tran)
    if debug:
        print(f"lr from power method: {lr}")
    (feature_dim, template_count) = W.size()
    (feature_dim_X, num_samples) = X.size()
    assert feature_dim == feature_dim_X
    prev_W = W
    
    next_W = W
    momentum = torch.ones(1, device=X.device)

    for n in range(max_iterations):
        cur_W = left_update_nmf_sgd(
                    X,
                    next_W,
                    H_tran,
                    learn_rate=lr,
                    shrinkage_sparsity=0,
                    min_val=min_val,
                    force_nonnegative=force_nonnegative,
            )
        next_W, momentum = fista_momentum(cur_W, prev_W, momentum)

        tolerance_achieved = relative_norm_difference(prev_W, cur_W)
        if n % 10 == 0 and debug:
            # print RMSE
            x_pred = cur_W @ H_tran.t()
            rmse = compute_rmse(X, x_pred)
            print(f"FISTA W iteration: {n} | RMSE: {rmse} | tolerance: {tolerance_achieved}")
        if tolerance is not None:
            if tolerance_achieved < tolerance:
                if debug:
                    print(f"FIST converged in {n+1} iterations.")
                break

        prev_W = cur_W

    if debug:
        print(f"tolerance achieved: {tolerance_achieved}")
    if return_iter_count:
        return cur_W, n+1
    else:
        return cur_W


###################################################################
# Optimizers




# This is the most current RMSProp optimizer
class RMSpropOptimizerCustom(object):
    """RMSprop optimizer.

    This has the same API as the PyTorch optimizers and can be used in place
    of them.

    It supports a tensor-valued learning rate, `lr`, which is supplied
    in the step(lr) method.

    Args:
        params: model parameters.

    """

    def __init__(self, params, alpha=0.99, epsilon=1e-8, default_lr = None):
        self.alpha = alpha
        self.epsilon = epsilon
        self.force_nonnegative = False
        self.weight_decay = None
        self.min_val = 0.0
        self.params = list(params)
        self.mean_square = [torch.zeros_like(param) for param in self.params]
        self.default_lr = default_lr


    def nonnegative_hook(self, min_val = 1e-5):
        """Clip weights to be non-negative.

        """
        self.force_nonnegative = True
        self.min_val = min_val


    def weight_decay_hook(self, decay):
        """Weight decay hook.

        Args:
            decay (float): Nonnegative weight decay amount.

        """
        self.weight_decay = decay

    def zero_grad(self):
        """
        Zeros out the gradients of the parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, lr = None):
        """
        Performs a single optimization step.
        """
        if lr is None:
            lr = self.default_lr
        with torch.no_grad(): # not strictly necessary
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    # Detach the gradient to avoid modifying it during the update
                    grad = param.grad.detach()
                    self.mean_square[i].mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
                    avg = self.mean_square[i].sqrt().add_(self.epsilon)
                    param.sub_(grad.div_(avg) * lr)

                    if self.weight_decay is not None:
                        param -= self.weight_decay*param

                    if self.force_nonnegative:
                        param[param < self.min_val] = self.min_val
                    


class RMSpropOptimizerSlidingWindow(object):
    """RMSprop optimizer with sliding learnable window.

    This has a sliding window of learnable weights.
    It is intended for positive factor networks (e.g., using PFC blocks) only.

    This has the same API as teh PyTorch optimizers and can be used in place
    of them.

    It supports a tensor-valued learning rate, `lr`, which is supplied
    in the step(lr) method.

    Args:
        params: model parameters.
        learnable_width (int): The width of the "unmasked" or "learnable window" subset
            of the weight matrix. The learnable window consists of all rows of the weight
            matrix and columns [i, i+learnable_width) where i is the current starting location
            of the left side of the sliding window.
        slide_speed (float): Advance the position of the learnable window by slide_speed columns
            on each call to step(). 

    """

    def __init__(self, params, alpha=0.99, epsilon=1e-8, default_lr = None,
                learnable_width=100, slide_speed=0.1,
                start_col_f = 0.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.force_nonnegative = False
        self.weight_decay = None
        self.min_val = 0.0
        self.params = list(params)
        # Check that all parameters are 2-dim matrices.
        for param in self.params:
            assert param.ndim == 2, "Param has bad size. Should be 2-dim."
        self.mean_square = [torch.zeros_like(param) for param in self.params]

        # Unmasking start index (float)
        self.start_col_f = start_col_f
        self.learnable_width = learnable_width
        self.slide_speed = slide_speed
        self.default_lr = default_lr


    def reset_learnable_window(self):
        self.start_col_f = 0.0

    def nonnegative_hook(self, min_val = 1e-5):
        """Clip weights to be non-negative.

        """
        self.force_nonnegative = True
        self.min_val = min_val


    def weight_decay_hook(self, decay):
        """Weight decay hook.

        Args:
            decay (float): Nonnegative weight decay amount.

        """
        self.weight_decay = decay

    def zero_grad(self):
        """
        Zeros out the gradients of the parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def get_learnable_window_borders(self):
        start_col = int(self.start_col_f)
        end_col = int(self.start_col_f + self.learnable_width)
        return start_col, end_col

    def step_without_update(self):
        """
        Advance the learnable window, but don't update any weights.
        """
        start_col = int(self.start_col_f)
        end_col = int(self.start_col_f + self.learnable_width)
        self.start_col_f += self.slide_speed


    def step(self, lr = None):
        """
        Performs a single optimization step.
        """
        start_col = int(self.start_col_f)
        end_col = int(self.start_col_f + self.learnable_width)
        self.start_col_f += self.slide_speed
        if lr is None:
            lr = self.default_lr
        with torch.no_grad():
            for i, param in enumerate(self.params):
                num_cols = param.shape[1]
                assert end_col <= num_cols, f"Insufficient number of columns in param: {param}"
                if param.grad is not None:
                    grad = param.grad.detach()

                    # Update the running mean square for only the unmasked region
                    self.mean_square[i][:, start_col:end_col].mul_(self.alpha).addcmul_(
                        grad[:, start_col:end_col], 
                        grad[:, start_col:end_col], 
                        value=1 - self.alpha
                    )
                    avg = self.mean_square[i].sqrt().add_(self.epsilon)

                    param_update = grad / avg * lr
                    param[:, start_col:end_col] -= param_update[:, start_col:end_col]

                    if self.weight_decay is not None:
                        param[:, start_col:end_col] -= self.weight_decay * param[:, start_col:end_col]

                    if self.force_nonnegative:
                        param[param < self.min_val] = self.min_val

                 
###################################################################
# Misc functions


def t2np(x):
    """Convert torch.Tensor (possibly on GPU) to a Numpy ndarray."""
    return x.to("cpu").numpy()


def np2t(x, device=None):
    """Convert a Numpy ndarray to a torch.Tensor (possibly on GPU)"""
    if device is not None:
        return torch.from_numpy(x).to(device)
    else:
        # This will probably return it on the cpu.
        return torch.from_numpy(x)

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

def relative_norm_difference(x1, x2, eps=1e-9):
    diff = torch.norm(x1 - x2, p=2)
    denom = torch.norm(x1,p=2)
    res = (diff + eps) / (denom + eps)
    return res
    

def hoyer_sparsity(x, dim=0):
    """Return the hoyer sparsity.

    Compute the sparseness measure from:
    "Non-negative Matrix Factorization with
    Sparseness Constraints"
    https://jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

    The sparsity is in the range [0, 1].
    If all elements are equal, then sparsity = 0.
    If one element is non-zero and all other elements are 0, then sparsity = 1.

    Args:
        dim (int): Dimension along with to compute the sparsity.

    """
    n_sqrt = math.sqrt(x.size(dim))
    norm_L1 = torch.norm(x, p=1, dim=dim)
    norm_L2 = torch.norm(x, p=2, dim=dim)
    sparsity = (n_sqrt - norm_L1 / norm_L2) / (n_sqrt - 1)
    return sparsity


def shrinkage_activation(x, lam=1e-3):
    """Group-shrinkage activation function.

    This is also known as the soft-thresholding operator which is used
    in ISTA.

    S_lambda(x) = sign(x)*max(|x| - lambda, 0)

    Args:
        x (Tensor): float tensor.

    Returns:
        (Tensor): float tensor of same size as x.

    """
    x1 = torch.abs(x) - lam
    z = torch.zeros_like(x)
    return torch.sign(x) * torch.maximum(x1, z)


class VogelSparseThreshold(torch.nn.Module):
    """My experimental sparse activation function.

    This version is initialized as a relu and gradually changes the threshold each iteration
    to target the keep fraction value using an internal SGD optimizer.
    """

    def __init__(self, keep_frac = 0.1, lr = 1e-3):
        super().__init__()
        self.keep_frac = keep_frac
        self.thresh = 0.0
        self.lr = lr
        self.cur_iter = 1

    def forward(self, x):
        x = F.threshold(x, self.thresh, 0.0)
        with torch.no_grad():
            nonzero_frac = torch.count_nonzero(x).item()/x.numel()
            if nonzero_frac > self.keep_frac:
                # Too many nonzero values are getting through. Increase threshold!
                self.thresh += self.lr
            elif nonzero_frac < self.keep_frac:
                self.thresh -= self.lr
        
        #if self.cur_iter % 10 == 0:
        #    print(f"thresh: {self.thresh} | nonzero_frac: {nonzero_frac}")
        self.cur_iter += 1
        return x

class LayerNormNoParams(nn.Module):
    """LayerNorm without params.

    This version does not currently have any learnable parameters.

    """

    def __init__(self, ndim):
        super().__init__()
        self.ndim = ndim

    def forward(self, input):
        return F.layer_norm(input, (self.ndim,), weight=None, bias=None, eps=1e-5)


def compute_rmse(x_target, x_pred):
    """Compute RMSE (root mean squared error)

    RMSE = sqrt of the mean of the sum of squared errors.

    Args:
        x_target (tensor): matrix of target values.
        x_pred (tensor): matrix of predicted values.

    Return:
        The RMSE as a scalar float value.

    """
    diff = x_target - x_pred
    sum_squares = (diff * diff).sum()
    denom = diff.numel()
    mean_square_err = sum_squares / denom
    ret = mean_square_err.sqrt()
    return ret


def check_rows(x: torch.Tensor, min_thresh: float = 2e-5) -> bool:
    """Raise an exception if x contains any rows such that all of its values
    are less than min_thresh.

    This is for debugging.
    """
    is_bad = torch.any(torch.all(x < min_thresh, dim=1))
    #assert is_bad == False, f"Found a row in the matrix with all values less than {min_thresh}"
    if is_bad:
        print(f"Warning: found a row in the matrix with all values less than {min_thresh}")

def test_check_rows():
    x = torch.rand(3,4)
    x[1,:] *= 0.01
    print(f"x:\n{x}")
    min_thresh = 0.01
    check_rows(x, min_thresh)
    
############################################################
# Normalization functions


def normalize_columns_at_most_equal_max_val(dest, source):
    """
    `source` and `dest` must have the same number of columns.
    For each column j, scale each column j of `dest` such
    that its new maximum value will be at most equal to the
    maximum value of the corresponding column j of `source`.
    For each column in `dest` that already has lower maximum value than the corresponding
    column in `source`, do nothing.

    Args:
        dest (torch.Tensor): M x N matrix. The modified version of this matrix
            will be returned.
        source (torch.Tensor): M x N matrix. This matrix will not be changed.

    Returns:
        dest (torch.Tensor): The modified dest tensor.
    """
    assert dest.shape[1] == source.shape[1], "source and dest must have the same number of columns"

    max_source = source.max(dim=0)[0]
    max_dest = dest.max(dim=0)[0]
    
    scale = torch.where(max_dest > max_source, max_source / max_dest, torch.tensor(1., device=dest.device))
    dest = dest * scale.view(1, -1)

    #min_val = 1e-5
    #dest = torch.clamp(dest, min=min_val) # optional

    return dest


def normalize_columns_equal_max_val(x1, x2, clamp_max_val=None):
    """
    x1 and x2 must have the same number of columns.
    For each column j, normalize the max value of
    x1[:, j] and x2[:, j] to have the same max values.

    Let's take max value as the mean of the max values of each column.
    Both columns are then scaled to have the new max value.

    Args:
        x1 (torch.Tensor): M x N matrix. 
        x2 (torch.Tensor): P x N matrix. 

    Returns:
        x1_out (torch.Tensor): The modified x1 tensor.
        x2_out (torch.Tensor): The modified x2 tensor.
    """
    assert x1.shape[1] == x2.shape[1], "x1 and x2 must have the same number of columns"

    x1_out = x1.clone()
    x2_out = x2.clone()

    max_x1 = x1_out.max(dim=0)[0]
    max_x2 = x2_out.max(dim=0)[0]

    max_new = 0.5 * (max_x1 + max_x2)
    
    if clamp_max_val is not None:
        # optional, limit the largest allowable value.
        max_new = torch.clamp(max_new, max=clamp_max_val) 

    scale1 = torch.where(max_x1 > 0, max_new / max_x1, torch.tensor(1., device=x1.device))
    scale2 = torch.where(max_x2 > 0, max_new / max_x2, torch.tensor(1., device=x2.device))

    x1_out *= scale1.view(1, -1)
    x2_out *= scale2.view(1, -1)

    return x1_out, x2_out


def test_normalize_columns_equal_max_val():
    print("test_normalize_columns_equal_max_val")
    x1 = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], device='cuda')
    x2 = torch.tensor([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]], device='cuda')
    print(f"x1:\n{x1}")
    print(f"x2:\n{x2}")

    x1_out, x2_out = normalize_columns_equal_max_val(x1, x2)
    print(f"x1_out:\n{x1_out}")
    print(f"x2_out:\n{x2_out}")

    # Calculate expected results
    max_x1 = x1.max(dim=0)[0]
    max_x2 = x2.max(dim=0)[0]
    max_new = 0.5 * (max_x1 + max_x2)
    scale1 = torch.where(max_x1 > 0, max_new / max_x1, torch.tensor(1., device=x1.device))
    scale2 = torch.where(max_x2 > 0, max_new / max_x2, torch.tensor(1., device=x2.device))
    expected_x1_out = x1 * scale1.view(1, -1)
    expected_x2_out = x2 * scale2.view(1, -1)

    assert torch.allclose(x1_out, expected_x1_out), f"Expected: {expected_x1_out}, but got: {x1_out}"
    assert torch.allclose(x2_out, expected_x2_out), f"Expected: {expected_x2_out}, but got: {x2_out}"
    print("PASS")




#############################################
# NMF functions

# Right updates:



# optimized for Numba CPU.
@njit(parallel=parallel_range, fastmath=use_fastmath)
def right_update_nmf_sgd_granular(
    X, W, H_tran, lr, weight_decay, sparsity_L1, min_val=1e-5
):
    """Perform the NMF inference update on H.

    This is parallelized over the column index of X so that update collisions
    are not possible.

    Note: H_tran = transpose of H is supplied instead of H.

    Note: H_tran is updated in place.

    This is the SGD algorithm from Kumozu Netflix prize example (NetflixRecSystem.cpp),
    modified to
    operate on dense matrices. It recomputes the reconstruction error before
    making each update.

    """
    (x_rows, x_cols) = X.shape
    (_, w_cols) = W.shape
    for j in prange(x_cols):
        for i in range(x_rows):
            val_X = X[i, j]
            # Compute estimate for val_X and the approximation error.
            approx_val_X = np.float32(0.0)
            k = 0
            while k < w_cols:
                approx_val_X += W[i, k] * H_tran[j, k]
                k += 1

            cur_error = val_X - approx_val_X
            k = 0
            while k < w_cols:
                h = H_tran[j, k]
                h = (
                    h
                    - lr * (W[i, k] * (-cur_error) + weight_decay * h)
                    - lr * sparsity_L1
                )
                if h < np.float32(min_val):
                    h = np.float32(min_val)
                H_tran[j, k] = h
                k += 1





def right_update_nmf_sgd(
    X,
    W,
    H_tran,
    learn_rate,
    shrinkage_sparsity=0,
    weight_decay=0,
    min_val=1e-5,
    force_nonnegative_H=True,
):
    """Apply right (inference) NMF SGD update under squared error loss.

    This is the standard NMF SGD update for H.

    Update H in
    X_target approx= W * H

    where X is the target value (i.e., X = X_target), and W is
    fixed and read-only by this function.
    That is, X_pred = W*H and so the error is X_pred - X:
    error = X_pred - X_target
          = W*H - X_target

    The inference update rule for the non-transposed udpate is:

    H <- H + learn_rate*(W_tran*X_target - W_tran*W*H)
    H <- relu(H)

    where grad_H = W_tran*W*H - W_tran*X_target
                 = W_tran(W*H - X_target)
                 = W_tran(X_pred - X_target)

    The inference update rule for the transposed update computed by
    this function is:
    (note: H' denotes tranpose of H)

    The following is the udpate (when `shrinkage_sparsity` regularizer is not used):

    H' <- relu(H' - learn_rate*(H'*W' - X')*W)

    Args:
        X (torch float tensor): shape is (x_feature_dim, example_count)
        W (torch float tensor): shape is (x_feature_dim, template_count)
        H_tran (torch float tensor): shape is (example_count, template_count)
        shrinkage_sparsity (float): An optional L1 sparsity regularizer. Set to 0 to disable.
        weight_decay (float): Optional weight decay.
        min_val (float): All values in the returned matrix will be clipped to have this as the
            minimum value (sometimes helps prevent numerical issues).

    """
    W_tran = W.transpose(1, 0)
    X_tran = X.transpose(1, 0)
    X_pred_tran = torch.matmul(H_tran, W_tran)
    error_tran = X_pred_tran - X_tran
    scaled_lr = learn_rate
    grad = torch.matmul(error_tran, W)
    H_tran = H_tran - learn_rate * grad
    if weight_decay > 0:
        H_tran = H_tran - scaled_lr*weight_decay*H_tran
    if shrinkage_sparsity > 0:
        # group-shrinkage activation function:
        H_tran = shrinkage_activation(H_tran, lam=shrinkage_sparsity)
    
    # Now clip all negative values in H to min_val.
    # ret = F.relu(H_tran) # old version
    if force_nonnegative_H:
        return torch.clamp(H_tran, min=min_val)
    else:
        return H_tran


def iterate_nmf_right_update(X, W, tolerance = 0.01, max_iterations=20, min_val = 1e-5,
                             return_iter_count = False, 
                             shrinkage_sparsity = 0, 
                             debug=False):
    """Iterate the NMF right update rule until approximate convergence.
    
    This runs several iterations of the SGD NMF right update rule to compute the update for H in

    X approx= W * H

    and returns the transpose of H.

    This function initializes H to zeros and then performs several iterations of inference
    (i.e., right update for H)
    until approximate convergence.

    By default, the learning rate is chosen automatically as 1/L where L is the largest
    eigenvalue in W^T*W. Note that this is the same learning rate that is used in FISTA.
    However, this function does not implement FISTA. Rather, it just implements the naive
    SGD updates.

    Args:
        tolerance: The iterations will end once the L2 norm between two consecutive
            H estimates drops below the specified tolerance. To disable, set to `None`.
        max_iterations: Maximum number of iterations to compute.
        min_val: Smallest allowable valeu in the returned H.

    """
    with torch.no_grad():
        lr = fista_compute_lr_from_weights(W)
    if debug:
        print(f"lr from power method: {lr}")
    (feature_dim, template_count) = W.size()
    (feature_dim_X, num_samples) = X.size()
    assert feature_dim == feature_dim_X
    prev_H_tran = torch.zeros((num_samples, template_count), dtype=X.dtype, device=X.device)

    for n in range(max_iterations):
        cur_H_tran = right_update_nmf_sgd(
                    X,
                    W,
                    prev_H_tran,
                    learn_rate=lr,
                    shrinkage_sparsity=shrinkage_sparsity,
                    force_nonnegative_H=True,
                )
        tolerance_achieved = relative_norm_difference(prev_H_tran, cur_H_tran)
        if n % 10 == 0 and debug:
            # print RMSE
            x_pred = W @ cur_H_tran.t()
            rmse = compute_rmse(X, x_pred)
            print(f"NMF iteration: {n} | RMSE: {rmse} | tolerance: {tolerance_achieved}")
        
        if tolerance is not None:
            if tolerance_achieved < tolerance:
                if debug:
                    print(f"Right update converged in {n+1} iterations.")
                break

        prev_H_tran = cur_H_tran
    if debug:
        print(f"tolerance achieved: {tolerance_achieved} | iteration: {n+1}")
    if return_iter_count:
        return cur_H_tran, n+1
    else:
        return cur_H_tran


def right_update_nmf_lee_seung(
    X, W, H_tran, lambda_1=0.0, lambda_2=0.0, epsilon=1e-5, min_val=1e-5
):
    """Apply NMF Lee-Seung right update rule.

    This version explicitly returns the updated H_tran = transpose(H).

    This version updates H_tran to minimize the Euclidean loss between X and W*H.

    Update H in
    X approx= W * H
    Recall that the usual update rules are:

                W'*X + epsilon
    H <- H .* ----------------
                W'*(W*H) + epsilon

    If using sparsity:

                W'*X + epsilon
    H <- H .* ------------------------------
                W'*(W*H) + lambda_1 + lambda_2*H + epsilon

    However, this function uses the transpose of H = H' (param `H_tran`) and so the corresponding
    updates become:

                  X'*W + epsilon
    H' <- H' * ---------------------------------------------
                (H'*W')*W + lambda_1 + lambda_2*H' + epsilon

    Args:

    X (torch float tensor): Input matrix of feature vectors. shape=(feature_dim, example_count)
    W (torch float tensor): The weights (basis vectors). shape=(feature_dim, template_count)
    H_tran (torch float tensor): The activations of the basis vectors with shape = (example_count, template_count). 
        This is H' which is the transpose of the usual H so that each column of H_tran activates the corresponding column
        of W.
    lambda_1 (float): L1 sparsity. Set to 0 to disable.
    lambda_2 (float): L2 sparsity. Set to 0 to disable.
    epsilon (float): small constant for numerical stability.

    Returns:
        The updated H_tran, with shape (example_count, template_count).

    """
    W_tran = W.t()
    X_tran = X.t()
    numer = X_tran @ W + epsilon
    X_pred_tran = H_tran @ W_tran
    denom = X_pred_tran @ W + epsilon
    if lambda_1 > 0:
        denom = denom + lambda_1
    if lambda_2 > 0:
        denom = denom + lambda_2 * H_tran
    H_scale = numer / denom
    H_ret = H_tran * H_scale
    # Optionally, keep all elements of H positive to prevent "dead" elements.
    H_ret = torch.clamp(H_ret, min=min_val)
    return H_ret


# in-place version (not differentiable)
def right_update_nmf_lee_seung_in_place(
    X, W, H_tran, lambda_1=0.0, lambda_2=0.0, epsilon=1e-5, min_val=1e-5
):
    W_tran = W.t()
    X_tran = X.t()
    numer = X_tran @ W + epsilon
    X_pred_tran = H_tran @ W_tran
    denom = X_pred_tran @ W + epsilon
    denom.add_(lambda_1)
    denom.add_(lambda_2 * H_tran)
    H_scale = numer / denom
    H_tran.mul_(H_scale)
    H_tran[H_tran < min_val] = min_val
    return H_tran

# Another in-place version
def right_update_nmf_lee_seung_in_place_einsum(
    X, W, H_tran, lambda_1=0.0, lambda_2=0.0, epsilon=1e-5, min_val=1e-5
):
    numer = torch.einsum('ji,jk->ik', X, W) + epsilon
    X_pred_tran = torch.einsum('ij,kj->ik', H_tran, W)
    denom = torch.einsum('ij,jk->ik', X_pred_tran, W) + epsilon
    denom.add_(lambda_1)
    denom.add_(lambda_2 * H_tran)
    H_scale = numer / denom
    H_tran.mul_(H_scale)
    H_tran.clamp_(min=min_val)
    return H_tran


# Left updates:



# Optimized for Numba CPU.
@njit(parallel=parallel_range, fastmath=use_fastmath)
def left_update_nmf_sgd_granular(
    X, W, H_tran, lr, weight_decay, sparsity_L1, min_val=1e-5
):
    """Perform the NMF learning update on W.

    This is parallelized over the row index of X so that update collisions are
    not possible.

    Note: H_tran = transpose of H is supplied instead of H.

    Note: W is updated in place.

    This is the SGD algorithm from Kumozu Netflix prize example (NetflixRecSystem.cpp),
    modified to
    operate on dense matrices. It recomputes the reconstruction error before
    making each update.

    """
    (x_rows, x_cols) = X.shape
    (_, w_cols) = W.shape
    # Optionally do in parallel
    for i in prange(x_rows):
        for j in range(x_cols):
            val_X = X[i, j]
            # Compute estimate for val_X and the approximation error.
            approx_val_X = np.float32(0.0)
            feature_ind = 0
            while feature_ind < w_cols:
                approx_val_X += W[i, feature_ind] * H_tran[j, feature_ind]
                feature_ind += 1

            cur_error = val_X - approx_val_X
            feature_ind = 0
            while feature_ind < w_cols:
                w = W[i, feature_ind]
                w = (
                    w
                    - lr * (H_tran[j, feature_ind] * (-cur_error) + weight_decay * w)
                    - lr * sparsity_L1
                )
                if w < np.float32(min_val):
                    w = np.float32(min_val)
                W[i, feature_ind] = w
                feature_ind += 1



def left_update_nmf_sgd(
    X,
    W,
    H_tran,
    learn_rate=1e-3,
    shrinkage_sparsity=0,
    weight_decay=0,
    min_val=1e-5,
    force_nonnegative=True,
    M = None,
):
    """Update W using the NMF left (learning) SGD update under squared error loss.

    Update W in the factorization:

    X approx= W * H

    where X is the target value (i.e., X = X_target), and H is
    fixed and read-only by this function.
    That is, X_pred = W*H and so the error is X_pred - X:
    error = X_pred - X
          = W*H - X

    The learning update rules is:

    W <- W + learn_rate*(X*H_tran - W*H*H_tran)

    where grad_W = W*H*H_tran - X*H_tran
                 = (W*H - X)*H_tran
                 = (X_pred - X)*H_tran

    If the masking matrix `M` is not None, it must be a float matrix of
    the same size as `X`. To mask an element of `X[i,j]`, `M[i,j]` should
    be set to 0. To keep the element, `M[i,j]` should be set to 1. Values
    in between 0 and 1 will interpolate between the original value and 0
    as the masking value varies betwen 1 and 0.
    When masking is enabled it simply modifies the error matrix above so that
    a masking value of 0 causes the corresponding masked error to be 0 as well:
    error_masked = error*M
    and `error_masked` is then used instead of `error` in the NMF update above.

    All matrices must be on the same device.

    Args:
        X (torch float tensor): shape is (x_feature_dim, example_count)
        W (torch float tensor): shape is (x_feature_dim, template_count)
        H (torch float tensor): shape is (template_count, example_count)
        M (torch float tensor): Masking matrix of same size as X. Must contain
            values in the range [0, 1]. A value of 0 indicates the corresponding element
            of `X` should be masked (i.e., ignroed) and a value of 1 indicates no masking.
            If `None`, masking is not enabled, which is equivalent to using a masking matrix
            of all ones.
        min_val (float): Small value to prevent numerical issues.
        force_nonnegative (bool): If True, constrain `W` to be non-negative. Otherwise, allow
            negative values, corresponding to semi-NMF.

    Returns:
        W

    """
    H = H_tran.transpose(1, 0)
    X_pred = torch.matmul(W, H)
    if M is None:
        error = X_pred - X
    else:
        error = M*(X_pred - X)
    
    scaled_lr = learn_rate    
    grad = torch.matmul(error, H_tran)
    W = W - learn_rate * grad
    if shrinkage_sparsity > 0:
        # group-shrinkage activation function:
        W = shrinkage_activation(W, lam=shrinkage_sparsity)
    if weight_decay > 0:
        W = W - scaled_lr*weight_decay*W
    if force_nonnegative:
        return torch.clamp(W, min=min_val)
    else:
        return W


# in-place version (not differentiable)
def left_update_nmf_lee_seung_(
    X,
    W,
    H_tran,
    lambda_1=0.0,
    lambda_2=0.0,
    learn_rate=1.0,
    epsilon=1e-5,
    min_val=1e-5,
):
    """Apply NMF Lee-Seung left update rule.

    Update W in
    X = W * H

    W is returned in place.

    The update rule is:

                 X*H' + epsilon
    W <- W .* ----------------------
                (W*H)*H' + epsilon

    If using sparsity:

                 X*H' + epsilon
    W <- W .* ----------------------
                (W*H)*H' + lambda_1 + lambda_2*H + epsilon


    Args:

    X (torch tensor): (read-only input) Input matrix of feature vectors. shape=(feature_dim, example_count)
    W (torch tensor): (read-write input/output) The weights (basis vectors). shape=(feature_dim, template_count).
        The result it also returned in W.
    H (torch tensor): (read-only input) The activations of the basis vectors. Each row is an activation
        vector.
        shape = (template_count, example_count).
    lambda_1: L1 sparsity regularization. Set to 0 to disable.
    lambda_2: L2 regularization (weight decay). Set to 0 to disable.
    learn_rate (float or tensor in [0, 1]). An optional learning rate. The learning rate. It must either be a scalar or a tensor of the
        wamte size as 'W' or an array of the same size as the number of basis vectors.

    """
    # H_tran = H.transpose(1, 0)
    H = H_tran.t()
    numer = X @ H_tran + epsilon
    X_approx = W @ H
    denom = X_approx @ H_tran + epsilon
    if lambda_1 > 0:
        denom = denom + lambda_1
    if lambda_2 > 0:
        denom = denom + lambda_2 * W
    W_scale = numer / denom
    assert W_scale.size() == W.size()  # debug
   
    if learn_rate < 1.0:
        # Take convex combination of scale factors and the ones matrix.
        W_scale = learn_rate * W_scale + (1 - learn_rate) * torch.ones_like(W)
        W[:] = W*W_scale
    else:
        W *= W_scale
    # Keep all elements at least min_val to prevent dead elements.
    W[W < min_val] = min_val

def left_update_nmf_lee_seung(
    X,
    W,
    H_tran,
    lambda_1=0.0,
    lambda_2=0.0,
    learn_rate=1.0,
    epsilon=1e-5,
    min_val=1e-5,
):
    """Apply NMF Lee-Seung left update rule.

    Update W in
    X = W * H

    The update rule is:

                 X*H' + epsilon
    W <- W .* ----------------------
                (W*H)*H' + epsilon

    If using sparsity:

                 X*H' + epsilon
    W <- W .* ----------------------
                (W*H)*H' + lambda_1 + lambda_2*H + epsilon


    Args:

    X (torch tensor): Input matrix of feature vectors. shape=(feature_dim, example_count)
    W (torch tensor): The weights (basis vectors). shape=(feature_dim, template_count).
        The result it also returned in W.
    H_tran (torch tensor): The activations of the basis vectors. This is the transpose of H.
        shape = (example_count, template_count).
    lambda_1: L1 sparsity regularization. Set to 0 to disable.
    lambda_2: L2 regularization (weight decay). Set to 0 to disable.
    learn_rate (float or tensor in [0, 1]). An optional learning rate. The learning rate. It must either be a scalar or a tensor of the
        wamte size as 'W' or an array of the same size as the number of basis vectors.

    Returns:
        W: Updated W.

    """
    H = H_tran.t()
    numer = X @ H_tran + epsilon
    X_approx = W @ H
    denom = X_approx @ H_tran + epsilon
    if lambda_1 > 0:
        denom = denom + lambda_1
    if lambda_2 > 0:
        denom = denom + lambda_2 * W
    W_scale = numer / denom
    assert W_scale.size() == W.size()  # debug
    if learn_rate < 1.0:
        # Take convex combination of scale factors and the ones matrix.
        W_scale = learn_rate * W_scale + (1 - learn_rate) * torch.ones_like(W)
        W = W*W_scale
    else:
        W = W*W_scale
    
    # Keep all elements at least min_val to prevent dead elements.
    return torch.clamp(W, min=min_val)


######################################################
# Data loaders

class MatrixBatcher2:
    """Return mini-batches from two underlying 2-dimensional matrices: `x`, `y`.

    This returns batches from an underlying dataset consisting of 2 matrices.
    The first matrix `x` contains the feature vectors as its columns so that
    the i'th column contains the feature for the i'th example. There are
    N columns in total. The same is true for `y`, which also contains
    N columns. The feature dimension could be different, though, so that
    the number of rows does not need to match between `x` and `y`.

    Note: If one of the matrices contains integer values (e.g., representing
    class labels), then the matrix size should be (1, num_examples).
    
    Args:
        x (torch.tensor of any type): Tensor of size (rows_x, N)
        y (torch.tensor of any type): Tensor of size (rows_y, N)
        batch_size (int): Number of samples per batch.
        shuffle (bool): If True, shuffles the columns before each epoch.
        use_pytorch_shape (bool): (default False) If False, return batches having
            shape (rows_x, batch_size) for the x batch and shape (rows_y, batch_size)
            for the y batch so that the feature vectors are placed in the columns of
            the batch tensors. Otherwise, if True, return the transpose to support
            the PyTorch convention that the feature vectors are the row vectors
            of the batch tensor.

    """

    def __init__(self, x, y, batch_size, shuffle=True, use_pytorch_shape = False):
        assert x.size(1) == y.size(1), "All input tensors should have same number of columns."
        self.x_dim = x.size(0)
        self.example_count = x.size(1)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        #torch.manual_seed(42)  # for reproducibility
        self.indices = torch.randperm(self.x.shape[1]) if self.shuffle else torch.arange(self.x.shape[1])
        self.current_idx = 0
        self.use_pytorch_shape = use_pytorch_shape

    def get_batch(self):
        """Get a batch of data.

        Selects `batch_size` columns from the 2 matrices and returns 
        the 2 corresponding sub-matrices as a tuple (x_batch, y_batch).
        The same columns are returned from each of the 2 matrices.
        If shuffle is True, permute the columns once per epoch (an epoch
        is N examples) so that randomly chosen columns are returned. Otherwise,
        start from column 0 and return consecutive columns, starting from column
        0 each epoch.

        Returns:
            (x_batch, y_batch): Tuple of torch tensors, each containing
                batch_size columns from the respective underlying matrices.
        """
        if self.current_idx + self.batch_size > self.x.shape[1]:
            if self.shuffle:
                #torch.random.shuffle(self.indices)
                self.indices = torch.randperm(self.x.shape[1])
            self.current_idx = 0

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        x_batch = self.x[:, batch_indices]
        y_batch = self.y[:, batch_indices]

        self.current_idx += self.batch_size

        if self.use_pytorch_shape:
            return x_batch.t(), y_batch.t()
        else:
            return x_batch, y_batch



def classification_dataset_to_batcher(dataset, batch_size, device, shuffle):
    """Convert a PyTorch classification dataset into a MatrixBatcher2.

    Given a PyTorch classification dataset, such as an instance of datasets.FashionMNIST,
    return a MatrixBatcher2 having the spcified batch size and move to `device`.
    """
    loader_pt = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    # Get tensors containing ALL of the examples:
    features, labels = next(iter(loader_pt))
    # Flatten images into 2D matrix of shape (num_pixels, num_examples) with one image per column vector.
    features = rearrange(features, 'b c h w -> (c h w) b')
    labels = rearrange(labels, 'b -> 1 b')
    features = features.to(device)
    labels = labels.to(device)
    return MatrixBatcher2(features, labels, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    test_check_rows()
    test_normalize_columns_equal_max_val()
    
    
    
