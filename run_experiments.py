
import os
import glob
import random
import sys
from collections import defaultdict

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from utils.utils import ValueAccumulator

from utils.utils import *
from models.models import *
import os

# Global settings:

datasets_root = "/home/vogel/datasets"
print(f"Using dataset root folder: {datasets_root}")

def sample_new_v1(num_slices):
    """Create the Kumozu deterministic sequence.

    Return a 4 x num_slices matrix containing the states.
    Each state is a 1-hot column vector in the returned matrix.

    Returns:
        torch.Tensor of the state sequence.

    """
    cur_state = 0
    states = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 1, 3, 2, 1]
    state_dim = 4
    out_mat = torch.zeros((state_dim, num_slices))
    for slice_ind in range(num_slices):
        state = states[cur_state]
        cur_state += 1
        cur_state = cur_state % len(states)
        out_mat[state, slice_ind] = 1.0

    return out_mat


def create_copy_task_dataset(rand_seq_len, pad_len, vocab_size, num_examples):
    """
    Create training and target sequences for the Copy Task.

    Generate a sequence of random integer values in [0, vocab_size-1]
    of length `rand_seq_len`. The generated sequence will be a sequence
    of 1-hot vectors of dimension `vocab_size`, not including the
    padding token and remember token which cannot appear in the random
    sequence. Thus, the total dimension of the generated sequence
    vectors is vocab_size + 2.
    The generated sequence is followed by `pad_len` "padding" tokens,
    which are represented as 1-hot vectors. This is followed by
    `rand_seq_len` "remember" tokens which are represented as 1-hot
    vectors. The random sequence must be recalled starting from the
    first "remember" token. The output token sequence therefore
    consists of rand_seq_len + pad_len 0-valued vectors followed by
    rand_seq_len 1-hot vectors which are identical to the first rand_seq_len
    1-hot vectors of the input sequence. Thus, the input sequence has
    dimension `vocab_size + 2` while the output (target) sequence has
    dimension `vocab_size` since it does not contain the "pad" or
    "remember" tokens.

    The output arrays are Numpy ndarrays of float32 type.

    Args:
        rand_seq_len (int): Length of the sequence to remember.
        pad_len (int): Number of padding tokens to place after
            the sequence to remember.
        vocab_size (int): The number of distinct random value that can
            potentially appear in the random sequence. This is the dimension
            of the 1-hot vectors in the random sequence.
        batch_size (int): batch size.

    Returns:
        (x, y): A tuple containing the input sequence x and the output sequence y.
            x has size (batch_size, sequence_length, in_dimension) where
            sequence_length = 2*rand_seq_len + pad_len. Note that
            in_dimension = vocab_size + 2.
            y has size (batch_size, sequence_length, out_dimension) where
            out_dimension = vocab_size.
    """
    # Dimensions of the input sequence.
    in_dimension = vocab_size + 2

    # Padding and remember tokens (one-hot vectors).
    pad_token = np.zeros(in_dimension)
    pad_token[-2] = 1

    # Output padding token.
    out_pad_token = np.zeros(vocab_size)

    # Define empty arrays to hold input and output data.
    x = np.zeros((num_examples, 2*rand_seq_len + pad_len, in_dimension), dtype=np.float32)
    y = np.zeros((num_examples, 2*rand_seq_len + pad_len, vocab_size), dtype=np.float32)

    # Generate batch.
    for i in range(num_examples):
        # Generate a random sequence.
        seq = np.eye(vocab_size)[np.random.choice(vocab_size, rand_seq_len)]
        # todo: hash the sequence so we can be sure that training sequences do not
        # appear in the validation/test set. Probably won't be a problem as long as
        # vocab_size and sequence length are long enough.

        # Create the input sequence.
        x[i, :rand_seq_len, :vocab_size] = seq
        x[i, rand_seq_len:rand_seq_len+pad_len, -2] = 1  # pad tokens
        x[i, rand_seq_len+pad_len:-rand_seq_len, -1] = 1  # remember tokens
        x[i, -rand_seq_len:, -1] = 1  # remember tokens

        # Create the output sequence.
        y[i, :rand_seq_len+pad_len, :] = out_pad_token
        y[i, rand_seq_len+pad_len:, :] = seq  # output sequence is identical to the input sequence

    return x, y


def test_create_copy_task_dataset():
    rand_seq_len = 5
    pad_len = 3
    vocab_size = 10
    num_examples = 4

    x, y = create_copy_task_dataset(rand_seq_len, pad_len, vocab_size, num_examples)

    assert x.shape == (num_examples, 2*rand_seq_len + pad_len, vocab_size + 2)
    assert y.shape == (num_examples, 2*rand_seq_len + pad_len, vocab_size)

    for i in range(num_examples):
        assert np.all(x[i, rand_seq_len:rand_seq_len+pad_len, -2] == 1)  # padding tokens
        assert np.all(x[i, rand_seq_len+pad_len:, -1] == 1)  # remember tokens
        assert np.all(y[i, :rand_seq_len+pad_len, :] == 0)  # output padding
        assert np.all(x[i, :rand_seq_len, :vocab_size] == y[i, rand_seq_len+pad_len:, :])  # copied sequence

    print(f"x:\n{x[0]}")
    print(f"y:\n{y[0]}")
    print("All tests passed!")
    

class RNNCopyTaskEvaluator:
    """Evaluate the forward pass of an RNN on the copy task."""

    def __init__(self, model, vocab_size, copy_seq_len, device) -> None:
        self.model = model
        self.vocab_size = vocab_size
        self.copy_seq_len = copy_seq_len
        self.h_dim = self.model.hidden_dim
        self.has_cell = model.has_cell
        # previous hidden state
        self.h_prev = None
        if self.has_cell:
            # previous cell state (if LSTM)
            self.c_prev = None
        self.device = device

    def forward(
        self,
        x_train,
        y_targets,
        is_reset,
        enable_backprop_through_time,
        loss_type="mse",
    ):
        """Evaluate the on a batch of inputs.

        Args:
            x_train (Tensor of float): Input features of shape (batch_size, feature_dim, seq_len).
            y_targets (Tensor of float or int): Target (ground truth) features of shape (batch_size, feature_dim, seq_len).
            is_reset (Tensor of boolean). Shape is (batch_size,) and is True if the i'th example
                in the batch is the start of a new sequence, in which case the internal state
                for this example needs to be reset.
        """
        (batch_size, feature_dim, seq_len) = x_train.size()

        if self.has_cell:
            c_zero = torch.zeros(batch_size, self.h_dim, device=x_train.device)
            if self.c_prev is None:
                self.c_prev = c_zero
            elif self.c_prev.size() != c_zero.size():
                print("c_prev size has changed. Resetting...")
                self.c_prev = c_zero
        
        h_zero = torch.zeros(batch_size, self.h_dim, device=x_train.device)
        if self.h_prev is None:
            self.h_prev = h_zero
        elif self.h_prev.size() != h_zero.size():
            print("h_prev size has changed. Resetting...")
            self.h_prev = h_zero

        # todo: reset state is not used in the Copy Task.
        # Reset the internal state of any examples in the batch where is_reset is True.
        is_reset_h = torch.unsqueeze(is_reset, 1)
        is_reset_h = is_reset_h.expand(batch_size, self.h_prev.size(1))
        self.h_prev = torch.where(is_reset_h, h_zero, self.h_prev)
        if self.has_cell:
            is_reset_c = torch.unsqueeze(is_reset, 1)
            is_reset_c = is_reset_h.expand(batch_size, self.c_prev.size(1))
            self.c_prev = torch.where(is_reset_c, c_zero, self.c_prev)

        model = self.model
        start_slice = seq_len - self.copy_seq_len
        computed_loss_count = 0
        for m in range(seq_len):
            x = x_train[:, :, m]
            if self.has_cell:
                y_t, c, h = model(x, self.c_prev, self.h_prev)
            else:
                y_t, h = model(x, self.h_prev)
            cur_targets = y_targets[:, :, m]
            # Only compute the loss on the last copy_seq_len slices.
            if m == start_slice:
                cur_targets_int = torch.argmax(cur_targets, dim=1)
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(y_t, cur_targets_int)
                    computed_loss_count = 1
                elif loss_type == "mse":
                    loss = F.mse_loss(y_t, cur_targets, reduction="sum")
                    computed_loss_count = 1
                else:
                    raise ValueError("Bad loss type")
                # compute accuracy:
                pred_int = torch.argmax(y_t, dim=1)
                batch_correct = pred_int.eq(cur_targets_int.view_as(pred_int)).sum().item()
                
            elif m > start_slice:
                cur_targets_int = torch.argmax(cur_targets, dim=1)
                if loss_type == "cross_entropy":
                    loss = loss + F.cross_entropy(y_t, cur_targets_int)
                    computed_loss_count += 1
                elif loss_type == "mse":
                    loss = loss + F.mse_loss(y_t, cur_targets, reduction="sum")
                    computed_loss_count += 1
                else:
                    raise ValueError("Bad loss type")
                # compute accuracy:
                pred_int = torch.argmax(y_t, dim=1)
                batch_correct += pred_int.eq(cur_targets_int.view_as(pred_int)).sum().item()
                
            if self.has_cell:
                if enable_backprop_through_time:
                    self.c_prev = c
                else:
                    self.c_prev = c.clone().detach()
            if enable_backprop_through_time:
                self.h_prev = h
            else:
                self.h_prev = h.clone().detach()

        # Must detach before returning because we can't let gradients flow between batches.
        if self.has_cell:
            self.c_prev = c.clone().detach()
        self.h_prev = h.clone().detach()

        assert computed_loss_count == self.copy_seq_len
        # Note: Just like a regular RNN, we only do 1 optimizer update per batch.
        if loss_type == "mse":
            loss = loss / feature_dim * self.copy_seq_len * batch_size
        elif loss_type == "cross_entropy":
            loss = loss / self.copy_seq_len

        acc_denom = batch_size*self.copy_seq_len
        return loss, batch_correct, acc_denom


class RNNSequentialClassificationEvaluator:
    """Evaluate the forward pass of an RNN.
    
    This evaluator is intended to be used for sequential classification tasks such
    as sequential MNIST.

    Format of inputs and targets:
    The input sequence should have shape (batch_size, seq_dim, sequence length).
    The for pixel-wise task, seq_dim=1, for row/column-wise, seq_dim=28 for MNIST,
    for example.
    The targets should have shape (batch_size,) and be of type integer so that there
    is an int-valued label associated with each example in the batch.
    
    Notes:
    - All examples in the batch are assumed to have the same sequence length.
    - The final time slice of the RNN computes the predicted class labels. The loss
    is computed only for this final slice. So, all previous outputs of the RNN are "don't care".

    """

    def __init__(self, model, device) -> None:
        self.model = model
        self.h_dim = self.model.hidden_dim
        # previous hidden state
        self.h_prev = None
        self.device = device
        

    @torch.no_grad()
    def print_stats(self, logger, do_plots=True, description=""):
        """Print debug info.

        """
        logger.debug(f"Model info for: {description}")
        self.model.print_stats(do_plots=do_plots, description=description)
        # self.h_tran has shape = (batch_size, seq_len, h_dim)
        # Since it is 3D, just pick one example to use for visualization:
        example_index = 3
        # h_prev shape = (h_dim, seq_len)
        h_prev = self.h_prev_all_slices[example_index, :, :]
        # h shape = (h_dim, seq_len)
        h = self.h_all_slices[example_index, :, :]
        # x shape = (x_dim, seq_len)
        x = self.x_targets_all_slices[example_index, :, :]
        logger.debug("H: min value: {}; max value: {}".format(h.min(), h.max()))
        h_sparsity = hoyer_sparsity(h, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        logger.debug(f"H: sparsity: {mean_sparsity}")
        
        if do_plots:
            config = self.model.config
            plot_image_matrix(h, file_name=f"{config.results_dir}/{description}_H.png", 
                                title=f"{description}: H")
            if config.rnn_type == "FactorizedRNN":
                W_h = self.model.W_h
                W_y = self.model.W_y
                W_x = self.model.W_x
            elif config.rnn_type == "vanillaRNN":
                W_h = self.model.h_prev_to_h.weight.data
                W_y = self.model.h_to_y.weight.data
                W_x = self.model.x_to_hidden_dim.weight.data.t()
            else:
                logger.debug("Current RNN not supported for plots")
                return
            W = torch.cat((W_y, W_h, W_x), dim=0)
            V_pred = W @ h
            plot_image_matrix(V_pred, file_name=f"{config.results_dir}/{description}_V_pred.png", 
                                title=f"{description}: V_pred")
            
            y_pred = self.y_pred_all_slices[example_index, :, :]
            (num_classes, seq_len) = y_pred.size()
            plot_image_matrix(y_pred, f"{config.results_dir}/{description}_y_pred.png", 
                title=f"{description}: Predicted values for training targets")
            
            h_prev_pred = W_h @ h
            plot_image_matrix(h_prev_pred, f"{config.results_dir}/{description}_h_prev_pred.png", 
                title=f"{description}: Predicted values for H_prev hidden states")
            
            x_pred = W_x @ h
            plot_image_matrix(x_pred, f"{config.results_dir}/{description}_x_pred.png", 
                title=f"{description}: Predicted values for X inputs")
            
            # Now plot the target values:
            # Plot the Y (class label) targets:
            y_targets = self.y_targets_last_slice[example_index]
            
            y_1hot_targets = F.one_hot(y_targets, num_classes).float()
            y_1hot_targets = y_1hot_targets.reshape(num_classes, 1)
           
            plot_image_matrix(y_1hot_targets, f"{config.results_dir}/{description}_Y_targets_1_example.png", 
                title=f"{description}: Y (class label) targets")

            plot_image_matrix(h_prev, f"{config.results_dir}/{description}_H_prev_targets_1_example.png", 
                title=f"{description}: H_prev (previous hidden states) targets")

            plot_image_matrix(x, f"{config.results_dir}/{description}_X_targets_1_example.png", 
                title=f"{description}: X (input features) targets")

    def forward(
        self,
        x,
        y_targets,
        enable_backprop_through_time,
        loss_type="mse",
    ):
        """Evaluate the on a batch of inputs.

        Batches should be supplied so that for each example in the batch, the first token/feature
        in the sequence immediately follows the last token of the previous batch.

        Args:
            x (Tensor of float): Input features of shape (batch_size, feature_dim, seq_len).
            y_targets (Tensor of int): Target (ground truth) features of shape (batch_size,) of int 
                containing the target class labels for the last time slice.
        """
        config = self.model.config
        if config.data_representaiton == "pixels":
            x = rearrange(x, 'b 1 h w -> b 1 (h w)')
        elif config.data_representaiton == "rows":
            x = rearrange(x, 'b c h w -> b h (c w)')
        elif config.data_representaiton == "exp":
            x = rearrange(x, 'b 1 h w -> b 1 (h w)')
            x = x.reshape((x.shape[0], x.shape[1]*config.in_dim, -1))
        else:
            raise ValueError('oops')
        (batch_size, feature_dim, seq_len) = x.size()
        device = x.device
        h_zero = torch.zeros(batch_size, self.h_dim, device=device)
        # For this task, the total training sequence fits within the batch examples, so
        # we always reset the state for each batch.
        self.h_prev = h_zero
        model = self.model
        # Create empty tensors for logging/visualization only:
        self.y_pred_all_slices = torch.zeros((batch_size, self.model.out_dim, seq_len), device=device)
        self.x_all_slices = torch.zeros((batch_size, feature_dim, seq_len), device=device)
        self.h_prev_all_slices = torch.zeros((batch_size, self.h_dim, seq_len), device=device)
        self.h_all_slices = torch.zeros((batch_size, self.h_dim, seq_len), device=device)

        with torch.no_grad():
            self.y_targets_last_slice = y_targets.clone().detach()
            self.x_targets_all_slices = x.clone().detach()

        loss_h = 0
        loss_x = 0
        loss_y = 0
        for m in range(seq_len):
            x_slice = x[:, :, m]  
            # y_t shape = (batch_size, y_dim)
            # h shape = (batch_size, h_dim)
            y_t, h = model(x_slice, self.h_prev)

            # logging
            with torch.no_grad():
                self.y_pred_all_slices[:, :, m] = y_t[:, :]
                self.h_prev_all_slices[:, :, m] = self.h_prev[:, :]
                self.h_all_slices[:, :, m] = h[:, :]

            # Only compute the loss on the last copy_seq_len slices.
            if m <= seq_len - 1:
                # Potentially compute the reconstruction loss over all time slices
                if loss_type == "mse_factorized":
                    h_prev_pred = torch.einsum("ij,kj->ki", model.W_h, h)
                    h_prev_targets = self.h_prev.clone().detach()
                    loss_h = loss_h + torch.nn.functional.mse_loss(h_prev_pred, h_prev_targets, reduction='sum')/torch.numel(h_prev_targets)

                    x_pred = torch.einsum("ij,kj->ki", model.W_x, h)
                    x_targets = x_slice.clone().detach()
                    loss_x = loss_x + torch.nn.functional.mse_loss(x_pred, x_targets, reduction='sum')/torch.numel(x_targets)

            if m == seq_len - 1:
                # We need to include the target prediction loss, which is computed only for the last time slice.
                if loss_type == "cross_entropy":
                    loss = F.cross_entropy(y_t, y_targets)
                elif loss_type == "mse_factorized" or loss_type == "mse":
                    (_, num_classes) = y_t.size()
                    y_1hot_targets = F.one_hot(y_targets, num_classes).float().clone().detach()
                    loss_y = torch.nn.functional.mse_loss(y_t, y_1hot_targets, reduction='sum')/torch.numel(y_1hot_targets)
                else:
                    raise ValueError("Bad loss type")
                # compute accuracy:
                pred_int = torch.argmax(y_t, dim=1)
                batch_correct = pred_int.eq(y_targets.view_as(pred_int)).sum().item()
            # h_prev = h.clone().detach()
            if enable_backprop_through_time:
                self.h_prev = h
            else:
                self.h_prev = h.clone().detach()

        # Must detach before returning because we can't let gradients flow between batches.
        self.h_prev = h.clone().detach()

        # Note: Just like a regular RNN, we only do 1 optimizer update per batch.
        if loss_type == "mse_factorized":
            loss = loss_y + loss_h/seq_len + loss_x/seq_len
        elif loss_type == "mse":
            loss = loss_y
        elif loss_type == "cross_entropy":
            pass
        else:
            raise ValueError("Bad loss type")

        acc_denom = batch_size
        return loss, batch_correct, acc_denom



def train_and_evaluate_copy_task_vanilla_rnn():
    """Evaluate vanilla RNNs on the well-known Copy Task.

    This uses conventional vanilla RNNs and supports the following:
    - BPTT: It can solve the task, e.g with padding length 5.
    - Without BPTT: It cannot solve the task for any padding length.

    LayerNorm is used on the hidden states, as we found it to improve performance.

    """
    config_experiment1 = AttributeDict(
        dataset = "copy task",
        enable_backprop_through_time = True,
        batch_size=512,
        train_iters=10000000,
        validation_iters=200,
        h_dim= 1024,
        use_optimizer = "rmsprop",
        device="cuda",
        learning_rate=5e-5,
        weight_decay=2e-4,
        loss_type = 'mse',
        #loss_type="cross_entropy",
        print_train_loss_every_iterations = 200,
        compute_validation_loss_every_iterations = 200,
        rnn_type = "vanillaRNN", #
        recurrent_drop_prob=0.0,
        input_drop_prob=0.0,
        rand_seq_len = 10,
        pad_len = 5,
        vocab_size = 10,
    )
    config = config_experiment1  
    config.seq_len = 2*config.rand_seq_len + config.pad_len
    
    rnn = VanillaRNN(config, config.vocab_size + 2, config.h_dim, config.vocab_size,
                    recurrent_drop_prob=config.recurrent_drop_prob,
                    input_drop_prob=config.input_drop_prob,
                    enable_input_layernorm=False,
                    enable_state_layernorm=True)
        
    rnn = rnn.to(config.device)
    #rnn = torch.compile(rnn)
    train_evaluator = RNNCopyTaskEvaluator(rnn, config.vocab_size, config.rand_seq_len, device=config.device)

    model_params = rnn.parameters()
    optimizer = RMSpropOptimizerCustom(model_params, default_lr=config.learning_rate)
    optimizer.weight_decay_hook(config.weight_decay)
    # train model
    rnn.train(True)
    best_validation_loss = None
    best_validation_accuracy = None
    best_epoch = 0
    train_loss_accumulator = ValueAccumulator()
    for n in range(config.train_iters):
        # x_train and y_targets both have shape = (batch_size, vocab_size, seq_len)
        
        is_reset = torch.ones(config.batch_size, dtype=torch.bool, device=config.device)
        x_train, y_targets = create_copy_task_dataset(config.rand_seq_len, 
                                                               config.pad_len, 
                                                               config.vocab_size, 
                                                               config.batch_size)
        x_train = np2t(x_train, device=config.device)
        x_train = rearrange(x_train, 'b l d -> b d l')
        y_targets = np2t(y_targets, device=config.device)
        y_targets = rearrange(y_targets, 'b l d -> b d l')
        
        loss, batch_correct, acc_denom  = train_evaluator.forward(
            x_train,
            y_targets,
            is_reset,
            enable_backprop_through_time=config.enable_backprop_through_time,
            loss_type=config.loss_type,
        )
        train_loss_accumulator.accumulate(loss.item())

        if n % config.print_train_loss_every_iterations == 0:
            print(f"iteration: {n} | training loss: {train_loss_accumulator.get_mean()}")
            train_loss_accumulator.reset()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if n % config.compute_validation_loss_every_iterations == 0:
            rnn.train(False)
            print("Computing validation loss...")
            val_evaluator = RNNCopyTaskEvaluator(rnn, config.vocab_size, config.rand_seq_len, device=config.device)
            val_acc_accumulator = ValueAccumulator()
            loss_val = 0
            val_iters = 0
            for m in range(config.validation_iters):
                is_reset_val = torch.ones(config.batch_size, dtype=torch.bool, device=config.device)
                x_val, y_targets_val = create_copy_task_dataset(config.rand_seq_len, 
                                                               config.pad_len, 
                                                               config.vocab_size, 
                                                               config.batch_size)
                # convert to Pytorch:
                x_val = np2t(x_val, device=config.device)
                x_val = rearrange(x_val, 'b l d -> b d l')
                y_targets_val = np2t(y_targets_val, device=config.device)
                y_targets_val = rearrange(y_targets_val, 'b l d -> b d l')
                
                with torch.no_grad():
                    loss, batch_correct, acc_denom = val_evaluator.forward(
                        x_val,
                        y_targets_val,
                        is_reset_val,
                        enable_backprop_through_time=False,
                        loss_type=config.loss_type,
                    )
                    loss_val += loss
                    val_acc_accumulator.accumulate(batch_correct, acc_denom)
                val_iters += 1

            rnn.train(True)
            validation_loss = loss_val.item()/val_iters
            if best_validation_loss is None:
                best_validation_loss = validation_loss
            elif validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            validation_accuracy = val_acc_accumulator.get_mean()
            if best_validation_accuracy is None:
                best_validation_accuracy = validation_accuracy
            elif validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
            print(f"validation loss: {validation_loss} | best loss so far: {best_validation_loss} | accuracy: {validation_accuracy} | best so far: {best_validation_accuracy}")
            print(f"numer: {val_acc_accumulator.get_total()} | denom: {val_acc_accumulator.get_count()}")

    print("Done!")



def train_and_evaluate_copy_task_factorized_rnn():
    """Evaluate the factorized RNN on the well-known Copy Task.

    This version uses the factorized RNN with conventional NMF updates for both W and H.

    It uses a mini-batched implementation.
    """
    config_experiment1 = AttributeDict(
        results_dir = "debug_plots",
        dataset = "copy task",
        batch_size=256,
        train_iters=10000000,
        validation_iters=100,
        h_dim=1024,
        nmf_inference_algorithm = "sequential_sgd",
        nmf_inference_iters_no_grad = 50,
        sparsity_L1_H = 0e-4,
        sparsity_L1_W = 0e-4,
        device="cuda",
        learning_rate=None,
        weight_decay_H=0e-5,
        weight_decay_W=0e-5,
        weights_noise_scale = 2e-2,
        enforce_nonneg_params = True, # It only works for non-negative params
        print_train_loss_every_iterations = 200,
        compute_validation_loss_every_iterations = 200,
        rand_seq_len = 10,
        pad_len = 5, # up to 10 works with perfect accuracy.
        vocab_size = 10,
    )

    config = config_experiment1
    logger = configure_logger()
    config.logger = logger
    network = None
    
    # train model
    best_validation_loss = None
    best_validation_accuracy = None
    train_loss_accumulator = ValueAccumulator()
    
    for n in range(config.train_iters):
        # x_train and y_targets both have shape = (batch_size, vocab_size, seq_len)
        x_train, y_targets = create_copy_task_dataset(config.rand_seq_len, 
                                                               config.pad_len, 
                                                               config.vocab_size, 
                                                               config.batch_size)
        x_train = np2t(x_train, device=config.device)
        x_train = rearrange(x_train, 'b l d -> b d l')
        y_targets = np2t(y_targets, device=config.device)
        y_targets = rearrange(y_targets, 'b l d -> b d l')
        if network is None:
            (batch_size, x_dim, seq_len) = x_train.size()
            config.x_dim = x_dim
            (_, y_dim, _) = y_targets.size()
            config.y_dim = y_dim
            config.basis_vector_count = config.h_dim
            config.seq_len = seq_len
            network = FactorizedRNNCopyTaskWithoutBackprop(config, config.device)
        
        y_pred, loss, batch_correct, acc_denom  = network.forward(
            x_train,
            y_targets
        )
        train_loss_accumulator.accumulate(loss.item())

        if n % config.print_train_loss_every_iterations == 0:
            config.logger.info(f"iteration: {n} | training loss: {train_loss_accumulator.get_mean()}")
            train_loss_accumulator.reset()

        network.update_weights()

        if n % config.compute_validation_loss_every_iterations == 0:
            config.logger.info("Computing validation loss...")
            val_acc_accumulator = ValueAccumulator()
            loss_val = 0
            val_iters = 0
            for m in range(config.validation_iters):
                x_val, y_targets_val = create_copy_task_dataset(config.rand_seq_len, 
                                                               config.pad_len, 
                                                               config.vocab_size, 
                                                               config.batch_size)
                # convert to Pytorch:
                x_val = np2t(x_val, device=config.device)
                x_val = rearrange(x_val, 'b l d -> b d l')
                y_targets_val = np2t(y_targets_val, device=config.device)
                y_targets_val = rearrange(y_targets_val, 'b l d -> b d l')
                
                with torch.no_grad():
                    y_pred, loss, batch_correct, acc_denom = network.forward(
                        x_val,
                        y_targets_val
                    )
                    loss_val += loss
                    val_acc_accumulator.accumulate(batch_correct, acc_denom)
                val_iters += 1

            validation_loss = loss_val.item()/val_iters
            if best_validation_loss is None:
                best_validation_loss = validation_loss
            elif validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            validation_accuracy = val_acc_accumulator.get_mean()
            if best_validation_accuracy is None:
                best_validation_accuracy = validation_accuracy
            elif validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
            config.logger.info(f"validation loss: {validation_loss} | best loss so far: {best_validation_loss} | mean per-slice accuracy: {validation_accuracy} | best so far: {best_validation_accuracy}")
            config.logger.info(f"numer: {val_acc_accumulator.get_total()} | denom: {val_acc_accumulator.get_count()}")
            network.print_stats(description="validation")


def run_sequential_mnist_rnn_experiments():
    """Run experiments on factorized and conventional RNNs on the Sequential MNIST task.

    This runs the sequential MNIST experiments used for the paper.
    
    """
    
    device = "cuda"

    config_experiment1 = AttributeDict(
        experiment_name = "Vanilla RNN, No BPTT",
        results_dir = "debug_plots",
        train_epoch_count=200,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 5,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        fista_tolerance = None,
        batch_size=200,
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=1e-5,
        weight_decay=1e-5,
        loss_type = "mse", # for standard RNNs and factorized RNNs
        compute_validation_loss_every_iterations = 200,
        rnn_type = "vanillaRNN", # works well, use dropout=0.1
        enable_rnn_state_layernorm = True, # For vanilla RNN. Set True for best results.
        recurrent_drop_prob=0.0, # Only for standard RNNs
        input_drop_prob=0.0, # Only for standard RNNs
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.8333

    # Try unrolling 20 iterations:
    config_experiment2 = AttributeDict(
        experiment_name = "Factorized RNN, No BPTT, Negative params",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 5,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=200, 
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 20, # only used by factorized RNN
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9428

    # Try unrolling 50 iterations:
    config_experiment3 = AttributeDict(
        experiment_name = "Factorized RNN, No BPTT, Negative params",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=200,
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 50, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9461

    # Set unrolled iteration count to 10 for this and the following experiments.
    config_experiment4 = AttributeDict(
        experiment_name = "Exp 4: Factorized RNN, No BPTT, negative params, h_dim=512, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100, # 100
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9623
    
    config_experiment5 = AttributeDict(
        experiment_name = "Exp 5: Factorized RNN, No BPTT, negative params, h_dim=2048, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9716
    # Notes: 
    # - accuracy is still good with only 10 unrolled iterations, no BPTT, allowing negative params.
    # - Took 53 epochs to train.

    config_experiment6 = AttributeDict(
        experiment_name = "Exp 6: Factorized RNN, No BPTT, Non-negative params, h_dim=2048, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = True, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.7581

    config_experiment7 = AttributeDict(
        experiment_name = "Exp 7: Factorized RNN, No BPTT, Non-negative params, h_dim=512, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = True, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.7743

    config_experiment8 = AttributeDict(
        experiment_name = "Exp 8: Factorized RNN, BPTT, negative params, h_dim=512, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100, # 100
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        nmf_inference_iters_no_grad = 0, # 0 # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9816

    config_experiment9 = AttributeDict(
        experiment_name = "Exp 9: Factorized RNN, BPTT, negative params, h_dim=2048, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # 0 # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.99

    config_experiment10 = AttributeDict(
        experiment_name = "Exp 10: Factorized RNN, BPTT, Non-negative params, h_dim=512, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # 0 # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = True, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9856

    config_experiment11 = AttributeDict(
        experiment_name = "Exp 11: Factorized RNN, BPTT, Non-negative params, h_dim=2048, 10 unroll iters",
        results_dir = "debug_plots",
        train_epoch_count=100,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 10,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        weights_noise_scale = 1e-2, # only used by factorized RNN
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=5e-5,
        weight_decay=1e-5,
        #loss_type = "mse", # for standard RNNs and factorized RNNs
        loss_type = "mse_factorized", # for factorized RNN
        compute_validation_loss_every_iterations = 200,
        nmf_inference_iters_no_grad = 0, # 0 # only used by factorized RNN
        nmf_gradient_iters_with_grad = 10, 
        learning_rate_H = None, # only used by factorized RNN
        sparsity_L1_H = 0, # only used by factorized RNN
        weight_decay_H = 0, # only used by factorized RNN
        rnn_type = "FactorizedRNN",
        dataset_name = "MNIST",
        enforce_nonneg_params = True, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9866

    # Train vanilla RNNs

    config_experiment12 = AttributeDict(
        experiment_name = "Exp 12: Vanilla RNN, No BPTT, h_dim=2048",
        results_dir = "debug_plots",
        train_epoch_count=200,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 5,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = False,
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=1e-5,
        weight_decay=1e-5,
        loss_type = "mse", # for standard RNNs and factorized RNNs
        rnn_type = "vanillaRNN",
        enable_rnn_state_layernorm = True, # For vanilla RNN. Set True for best results.
        recurrent_drop_prob=0.0, # Only for standard RNNs
        input_drop_prob=0.0, # Only for standard RNNs
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9406

    config_experiment13 = AttributeDict(
        experiment_name = "Exp 13: Vanilla RNN, BPTT, h_dim=512",
        results_dir = "debug_plots",
        train_epoch_count=200,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 5,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=512,
        device=device,
        learning_rate=1e-5,
        weight_decay=1e-5,
        loss_type = "mse", # for standard RNNs and factorized RNNs
        rnn_type = "vanillaRNN",
        enable_rnn_state_layernorm = True, # For vanilla RNN. Set True for best results.
        recurrent_drop_prob=0.0, # Only for standard RNNs
        input_drop_prob=0.0, # Only for standard RNNs
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9794

    config_experiment14 = AttributeDict(
        experiment_name = "Exp 14: Vanilla RNN, BPTT, h_dim=2048",
        results_dir = "debug_plots",
        train_epoch_count=200,
        # Fraction of the training set to use for the train split. Rest will be used for validation split.
        train_split_fraction = 0.85,
        early_stopping_max_no_loss_improvement_epochs = 5,
        save_model_params_file = 'saved_models/network_parameters.pth',
        enable_backprop_through_time = True,
        fista_tolerance = None,
        batch_size=100,
        train_iters=10000000,
        h_dim=2048,
        device=device,
        learning_rate=1e-5,
        weight_decay=1e-5,
        loss_type = "mse", # for standard RNNs and factorized RNNs
        rnn_type = "vanillaRNN",
        enable_rnn_state_layernorm = True, # For vanilla RNN. Set True for best results.
        recurrent_drop_prob=0.0, # Only for standard RNNs
        input_drop_prob=0.0, # Only for standard RNNs
        dataset_name = "MNIST",
        enforce_nonneg_params = False, # only used by factorized RNN
        data_representaiton = "rows",
    )
    # Test set accuracy: 0.9866
    
    # Decide which configs to run by adding them to the config list.
    config_list = [config_experiment1, config_experiment4, config_experiment5, config_experiment6, config_experiment7, config_experiment8, config_experiment9, config_experiment10, config_experiment11, config_experiment12, config_experiment13, config_experiment14]

    logger = configure_logger()
    for config in config_list:
        logger.info("----------------------------------------------------")
        logger.info(f"Running experiment: {config.experiment_name}")
        logger.info(f"Using config:\n{config}")
        config.logger = logger
        sequential_mnist_task_various_rnns_batched_loader(config)
        logger.info(f"Test set accuracy: {config.test_accuracy}")
        logger.info(f"Test set loss: {config.test_loss}")
        logger.info("----------------------------------------------------")


def sequential_mnist_task_various_rnns_batched_loader(config):
    """Evaluate factorized and conventional RNNs on the well-known sequential MNIST task.

    Fashion MNIST and CIFAR10 are also supported.

    This supports both with and without BPTT.
    """
    def evaluate_rnn(config, network, batch_loader):
        network.train(False)
        evaluator = RNNSequentialClassificationEvaluator(rnn, device=config.device)
        acc_accumulator = ValueAccumulator()

        loss_sum = 0
        iters = 0
        for (data, y_targets) in batch_loader:
            sys.stdout.write(".")
            sys.stdout.flush()
            data = data.to(config.device)
            y_targets = y_targets.to(config.device)
                
            with torch.no_grad():
                loss, batch_correct, acc_denom = evaluator.forward(
                    data,
                    y_targets,
                    enable_backprop_through_time=False,
                    loss_type=config.loss_type,
                )
                loss_sum += loss
                acc_accumulator.accumulate(batch_correct, acc_denom)
            iters += 1
        sys.stdout.write("\n")
        sys.stdout.flush()
        network.train(True)
        mean_loss = loss_sum.item()/iters
        accuracy = acc_accumulator.get_mean()
        return mean_loss, accuracy, evaluator

    if config.dataset_name == "MNIST":
        # Load the MNIST dataset
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        image_side_pixels = 28
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == "Fashion MNIST":
        # Load the Fashion MNIST dataset
        dataset_train_full = datasets.FashionMNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        image_side_pixels = 28
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'CIFAR10':
        # Load the CIFAR10 dataset
        dataset_train_full = datasets.CIFAR10(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.CIFAR10(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        image_side_pixels = 32
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'CIFAR100':
        # Load the CIFAR100 dataset
        dataset_train_full = datasets.CIFAR100(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.CIFAR100(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 100 # number of classes
        image_side_pixels = 32
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    # Get the data tensors containing the examples:
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

    if config.data_representaiton == "pixels":
        config.in_dim = 1
    elif config.data_representaiton == "rows":
        config.in_dim = image_side_pixels
    if config.rnn_type == "vanillaRNN":
        rnn = VanillaRNN(config, config.in_dim, config.h_dim, num_classes,
                         recurrent_drop_prob=config.recurrent_drop_prob,
                            input_drop_prob=config.input_drop_prob,
                            enable_bias = True,
                            enable_input_layernorm=False,
                            enable_state_layernorm=config.enable_rnn_state_layernorm,
                            no_params_layer_norm=False)
    elif config.rnn_type == "FactorizedRNN":
        rnn = FactorizedRNN(config, config.in_dim, config.h_dim, num_classes)
    rnn = rnn.to(config.device)
    #rnn = torch.compile(rnn)
    
    train_evaluator = RNNSequentialClassificationEvaluator(rnn, device=config.device)
    model_params = rnn.parameters()
    optimizer = RMSpropOptimizerCustom(model_params, default_lr=config.learning_rate)
    optimizer.weight_decay_hook(config.weight_decay)
    
    # train model
    rnn.train(True)
    best_validation_loss = None
    best_validation_accuracy = None
    best_epoch = 0
    no_loss_improvement_epochs = 0
    early_stop = False
    train_loss_accumulator = ValueAccumulator()
    for epoch in range(config.train_epoch_count):
        if early_stop:
            break
        for (x_train, y_targets) in train_loader:
            sys.stdout.write(".")
            sys.stdout.flush()
            # x_train has shape (batch_size, 1, height, width).
            x_train = x_train.to(config.device)
            y_targets = y_targets.to(config.device)
            
            loss, batch_correct, acc_denom  = train_evaluator.forward(
                x_train,
                y_targets,
                enable_backprop_through_time=config.enable_backprop_through_time,
                loss_type=config.loss_type,
            )
            train_loss_accumulator.accumulate(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if config.enforce_nonneg_params:
                rnn.clip_weights_nonnegative()
        sys.stdout.write("\n")
        sys.stdout.flush()
        train_loss = train_loss_accumulator.get_mean()
        config.logger.debug(f"epoch: {epoch} | training loss: {train_loss}")
        train_loss_accumulator.reset()    

        config.logger.debug('Evaluating model on validation dataset...')
        validation_loss, validation_accuracy, val_evaluator = evaluate_rnn(config, rnn, val_loader)
        if best_validation_accuracy is None:
            best_validation_accuracy = validation_accuracy
        elif validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

        if (best_validation_loss is None) or (validation_loss < best_validation_loss):
            best_validation_loss = validation_loss
            best_epoch = epoch
            config.logger.debug("Saving the model parameters.")
            no_loss_improvement_epochs = 0
            model_save = {
                'model_state_dict': rnn.state_dict(),
                #'train_acc': train_accuracy,
                'train_loss': train_loss,
                'validation_acc': validation_accuracy,
                'validation_loss': validation_loss,
                'epoch': epoch,
            }
            torch.save(model_save, config.save_model_params_file)
        else:
            no_loss_improvement_epochs += 1
            if no_loss_improvement_epochs == config.early_stopping_max_no_loss_improvement_epochs:
                early_stop = True
                config.logger.debug(f"Early stopping threshold of {config.early_stopping_max_no_loss_improvement_epochs} epochs without andy loss improvement. Stopping training.")
            config.logger.debug(f"Previous better validation loss: {best_validation_loss} and accuracy: {best_validation_accuracy} | occured epoch: {best_epoch} | epochs without improvement: {no_loss_improvement_epochs}")
        
        config.logger.debug(f"validation loss: {validation_loss} | best loss so far: {best_validation_loss} | accuracy: {validation_accuracy} | best so far: {best_validation_accuracy}")
        if True:
            # debug plots
            val_evaluator.print_stats(config.logger, do_plots=True, description="validation")

    # Evaluate best model on test set:
    config.logger.debug('Evaluating model on test dataset...')
    config.logger.debug("Loading the saved model parameters.")
    model_load = torch.load(config.save_model_params_file)
    config.logger.debug("Loading saved model parameters.")
    rnn.load_state_dict(model_load['model_state_dict'])
    train_loss = model_load['train_loss']
    validation_acc = model_load['validation_acc']
    train_epochs = model_load['epoch']
    config.logger.debug(f"Saved model: Training loss: {train_loss}")
    config.logger.debug(f"Saved model: Best validation accuracy during training: {validation_acc}")
    config.logger.debug(f"Saved model: Training epochs (at early stopping point): {train_epochs}")

    config.logger.debug('Evaluating model on validation dataset...')
    test_loss, test_accuracy, test_evaluator = evaluate_rnn(config, rnn, test_loader)
    config.logger.debug(f"Test loss: {test_loss}")
    config.logger.debug(f"Test accuracy: {test_accuracy}")
    config.test_loss = test_loss
    config.test_accuracy = test_accuracy



def sequential_mnist_factorized_rnn_conventional_nmf():
    """Evaluate factorized RNN on the sequential MNIST task using conventional NMF W and H updates.

    Evaluate the factorized RNN on the Sequential MNIST classification task.

    This version uses the factorized RNN with conventional NMF updates for both W and H. This is
    similar to the inference and learning algorithms used in the RNNs in the "positive factor networks" paper.

    """

    config = AttributeDict(
        results_dir = "debug_plots",
        fista_tolerance = None,
        z_mask_val = 1.0,
        dataset = "MNIST",
        data_representaiton = "rows",
        batch_size=200,
        train_epochs=10000000,
        validation_iters=100,
        basis_vector_count = 512,
        nmf_inference_algorithm = "fista",
        learning_algorithm = "sgd",
        nmf_inference_iters_no_grad = 25,
        sparsity_L1_H = 0e-4,
        sparsity_L1_W = 0e-4,
        device="cuda",
        learning_rate_H = None,
        learning_rate=None,
        weight_decay_H=0e-5,
        weight_decay_W=0e-5,
        weights_noise_scale = 1e-2,
        enforce_nonneg_params = True,
        print_train_loss_every_iterations = 20,
        compute_validation_loss_every_iterations = 200,
    )
    if config.dataset == "MNIST":
        # Load the MNIST dataset
        dataset_train = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        config.image_side_pixels = 28
        print('Training on MNIST dataset.')
    elif config.dataset == "Fashion MNIST":
        # Load the Fashion MNIST dataset
        dataset_train = datasets.FashionMNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        config.image_side_pixels = 28
        print('Training on Fashion MNIST dataset.')
    elif config.dataset == 'CIFAR10':
        # Load the CIFAR10 dataset
        dataset_train = datasets.CIFAR10(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.CIFAR10(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        config.image_side_pixels = 32
        print('Training on CIFAR10 dataset.')
    elif config.dataset == 'CIFAR100':
        # Load the CIFAR100 dataset
        dataset_train = datasets.CIFAR100(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.CIFAR100(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 100 # number of classes
        config.image_side_pixels = 32
        print('Training on CIFAR100 dataset.')
    # Get the data tensors containing the examples:
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    config.y_dim = num_classes
    network = FactorizedRNNWithoutBackpropSeqMNIST(config, config.device)
    
    # train model
    best_validation_loss = None
    best_validation_accuracy = None
    train_loss_accumulator = ValueAccumulator()
    
    for epoch in range(config.train_epochs):
        n = 0
        for (x_train, y_targets_train) in train_loader:
            x_train = x_train.to(config.device)
            y_targets_train = y_targets_train.to(config.device)
                    
            y_pred, loss, batch_correct, acc_denom  = network.forward(
                x_train,
                y_targets_train
            )
            train_loss_accumulator.accumulate(loss.item())

            if n % config.print_train_loss_every_iterations == 0:
                print(f"epoch: {epoch} | iteration: {n} | training loss: {train_loss_accumulator.get_mean()}")
                network.print_stats(do_plots=False, description="train")
                train_loss_accumulator.reset()

            network.update_weights()

            if n % config.compute_validation_loss_every_iterations == 0:
                print("Computing validation loss...")
                val_acc_accumulator = ValueAccumulator()
                loss_val = 0
                val_iters = 0
                for (x_val, y_targets_val) in test_loader:
                    x_val = x_val.to(config.device)
                    y_targets_val = y_targets_val.to(config.device)
                    y_pred, loss, batch_correct, acc_denom = network.forward(
                        x_val,
                        y_targets_val
                    )
                    loss_val += loss
                    val_acc_accumulator.accumulate(batch_correct, acc_denom)
                    val_iters += 1

                validation_loss = loss_val.item()/val_iters
                if best_validation_loss is None:
                    best_validation_loss = validation_loss
                elif validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                validation_accuracy = val_acc_accumulator.get_mean()
                if best_validation_accuracy is None:
                    best_validation_accuracy = validation_accuracy
                elif validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                print(f"validation loss: {validation_loss} | best loss so far: {best_validation_loss} | accuracy: {validation_accuracy} | best so far: {best_validation_accuracy}")
                print(f"numer: {val_acc_accumulator.get_total()} | denom: {val_acc_accumulator.get_count()}")
                network.print_stats(description="validation")
            n += 1
    print("Done!")


def run_factorized_or_mlp_classifier(config):
    """Train and/or evaluate PFC-based or MLP classifier.

    
    """
    config.logger.debug(f"Running experiment: {config.experiment_name}")

    # Filter the data to only include specific digits and return
    # the dataset.
    # This is for the continual learning experiments.
    def filter_dataset_labels(data, include_labels):
        filtered_images = []
        filtered_labels = []
        for image, label in data:
            if label in include_labels:
                filtered_images.append(image)
                filtered_labels.append(label)
        return TensorDataset(torch.stack(filtered_images), torch.tensor(filtered_labels))

    @torch.no_grad()
    def evaluate_model(config, network, data_batcher, num_classes):
        loss_accumulator = ValueAccumulator()
        accuracy_accumulator = ValueAccumulator()
        network.train(False)
        for _ in range(data_batcher.example_count // data_batcher.batch_size):
            x, y = data_batcher.get_batch()
            sys.stdout.write(".")
            sys.stdout.flush()
            
            if config.network_type == "MLP":
                # It takes the batch size as the 1st dimension.
                y_1hot = F.one_hot(y, num_classes).squeeze().float()
                x = x.t() # put batch size in 1st dimension.
                y_pred, loss = network(x, y_1hot)
                y_pred = y_pred.t()
            elif config.network_type == "PFCLayer" or config.network_type == "PFCLayer2":
                # It takes the batch size as the 2nd dimension.
                y_1hot = F.one_hot(y, num_classes).squeeze().t().float()
                y_pred, loss = network(x, y_1hot)
            else:
                raise ValueError("bad value")
                    
            loss_accumulator.accumulate(loss.item())
            predicted_int = y_pred.argmax(dim=0, keepdim=True)
            batch_size = predicted_int.size(1)
            batch_correct = predicted_int.eq(y).sum().item()
            accuracy_accumulator.accumulate(batch_correct, batch_size)
            
        mean_loss = loss_accumulator.get_mean()
        accuracy = accuracy_accumulator.get_mean()
        network.train(True)
        print("")
        return mean_loss, accuracy
    
    if config.dataset_name == 'MNIST':
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #input_feature_dim = 28*28
        num_classes = 10 # number of classes
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'Fashion MNIST':
        dataset_train_full = datasets.FashionMNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #input_feature_dim = 28*28
        num_classes = 10 # number of classes
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'CIFAR10':
        dataset_train_full = datasets.CIFAR10(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        
        dataset_test = datasets.CIFAR10(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 10 # number of classes
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'CIFAR100':
        dataset_train_full = datasets.CIFAR100(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.CIFAR100(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        num_classes = 100 # number of classes
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'split MNIST':
        # Train on the 2 digits in the current split. Validate on on the current split and all earlier splits.
        # Evaluate the test set on all earlier splits including the current one.
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #print(f"Size of the full training dataset: {len(dataset_train_full)}")
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_train_full = filter_dataset_labels(dataset_train_full, config.split)
        dataset_test = filter_dataset_labels(dataset_test, config.all_seen_classes)
        num_classes = 10 # number of classes

        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])

        # Now add the validation dataset examples from the previous split, if available.
        is_all_seen_same_as_current = list(config.split) == config.all_seen_classes
        if os.path.exists(config.save_model_params_file) and not is_all_seen_same_as_current:
            model_load = torch.load(config.save_model_params_file)
            if 'validation_dataset' in model_load:
                config.logger.debug("Adding previous validation splits to current validation split.")
                previous_validation_dataset = model_load['validation_dataset']
                config.logger.debug(f"Current validation split for {config.split} contains {len(dataset_val)} examples.")
                config.logger.debug(f"Previous validation splits contain {len(previous_validation_dataset)} examples.")
                dataset_val = ConcatDataset([dataset_val, previous_validation_dataset])
                config.logger.debug(f"New combined validation split contains {len(dataset_val)} examples.")

    elif config.dataset_name == 'OOD MNIST':
        # Train on a subset of the available classes but evaluate on all classes.
        # The classes not trained on will be considered OOD at evaluation time.
        # Train on the first 5 digits only (for train split and validation split)
        train_labels = [0, 1, 2, 3, 4]
        # Evaluate on on all digits, so that digits 5-9 are considered OOD examples. So, we just
        # use the full test set as usual.
        
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #print(f"Size of the full training dataset: {len(dataset_train_full)}")
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_train_full = filter_dataset_labels(dataset_train_full, train_labels)
        
        # Note: The network will only use 5 class labels because it is only trained on 5 digit classes.
        num_classes = len(train_labels)

        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'OOD MNIST/Fashion MNIST':
        # Train on MNIST digits and evaluate on MNIST and Fashion MNIST so that the
        # Fashion MNIST images are OOD.
        # Note: validation split contains same distribution as training split (i.e., only
        # MNIST imiages) so that we can use early stopping.
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #print(f"Size of the full training dataset: {len(dataset_train_full)}")
        dataset_test_mnist = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test_fashion_mnist = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        
        num_classes = 10
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
        # Now combine the two test datasets into 1.
        test_images = []
        test_labels = []
        for image, label in dataset_test_mnist:
            # Add the MNIST test examples as is.
            test_images.append(image)
            test_labels.append(label)
        for image, label in dataset_test_fashion_mnist:
            # Add the Fashion MNIST test example, but modify its label to be +10 to avaoid overlap with MNIST.
            # The total range of test labels in the combined dataset will be in [0, 19].
            test_images.append(image)
            new_label = label + 10
            test_labels.append(new_label)
        dataset_test = TensorDataset(torch.stack(test_images), torch.tensor(test_labels))
    elif config.dataset_name == 'OOD Fashion MNIST':
        # Train on a subset of the available classes but evaluate on all classes.
        # The classes not trained on will be considered OOD at evaluation time.
        # Train on the first 5 digits only (for train split and validation split)
        train_labels = [0, 1, 2, 3, 4]
        # Evaluate on on all digits, so that digits 5-9 are considered OOD examples. So, we just
        # use the full test set as usual.
        dataset_train_full = datasets.FashionMNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_train_full = filter_dataset_labels(dataset_train_full, train_labels)
        
        # Note: The network will only use 5 class labels because it is only trained on 5 digit classes.
        num_classes = len(train_labels)

        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
    elif config.dataset_name == 'ordered MNIST':
        # Create a training dataset in which we present the examples in order their sorted labels.
        # It also supports an optional fraction of examples to retain for each class in keep_frac.
        dataset_train_full = datasets.MNIST(datasets_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        dataset_test = datasets.MNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        #input_feature_dim = 28*28
        num_classes = 10 # number of classes
        # Split training set into train and validation splits.
        train_split_fraction = config.train_split_fraction
        train_size = int(train_split_fraction * len(dataset_train_full))
        val_size = len(dataset_train_full) - train_size
        
        dataset_train, dataset_val = random_split(dataset_train_full, [train_size, val_size])
        #print(f"Number of training split examples in the train split before creating class imbalance: {len(dataset_train)}")
        # We want to keep the validation split as usual.
        keep_fractions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        def filter_dataset_labels(dataset, keep_fractions, shuffle=config.ordered_shuffle, seed=42):
            random.seed(seed)
            torch.manual_seed(seed)

            # Group indices by their class label.
            indices_by_class = defaultdict(list)
            for idx, (_, label) in enumerate(dataset):
                indices_by_class[label].append(idx)

            # Calculate how many samples to keep for each class.
            keep_indices = []
            for label, fraction in enumerate(keep_fractions):
                indices = indices_by_class[label]
                n_keep = int(len(indices) * fraction)
        
                # Randomly choose the indices to keep.
                keep_indices.extend(sorted(random.sample(indices, n_keep)))

            if shuffle:
                # Shuffle keep_indices
                random.shuffle(keep_indices)

            return torch.utils.data.Subset(dataset, keep_indices)

        dataset_train = filter_dataset_labels(dataset_train, keep_fractions)
        #print(f"Number of training split examples left in the train split after creating class imbalance: {len(dataset_train)}")
    else:
        raise ValueError("Bad value")

    if config.experiment_name == "Deterministic learnable window" or config.experiment_name == "Unlearning":
        shuffle_train = False
    else:
        shuffle_train = True
        
    train_batcher = classification_dataset_to_batcher(dataset_train, config.batch_size, config.device, shuffle=shuffle_train)
    x_dim = train_batcher.x_dim
    num_slices = train_batcher.example_count
    val_batcher = classification_dataset_to_batcher(dataset_val, config.batch_size, config.device, shuffle=False)
    test_batcher = classification_dataset_to_batcher(dataset_test, config.batch_size, config.device, shuffle=False)
    
    # Select model
    if config.network_type == "MLP":
        config.config_MLP.input_dim = x_dim
        config.config_MLP.output_dim = num_classes
        network = MLP(config.config_MLP)
    elif config.network_type == "PFCLayer":
        config.config_PFCBlock.x_dim = x_dim
        config.config_PFCBlock.y_dim = num_classes
        network = PFCBlock(config.config_PFCBlock)
    elif config.network_type == "PFCLayer2":
        config.config_PFC2Block.x_dim = x_dim
        config.config_PFC2Block.hidden_dim = x_dim
        config.config_PFC2Block.y_dim = num_classes
        network = PFC2Layer(config.config_PFC2Block)
    else:
        raise ValueError("bad value")
    network = network.to(config.device)

    if config.resume_training_from_checkpoint and config.run_training:
        config.logger.debug('Not training from scratch. Resuming training from model checkpoint file...')
        config.logger.debug("Loading the saved model parameters.")
        model_load = torch.load(config.save_model_params_file)
        config.logger.debug("Loading saved model parameters.")
        network.load_state_dict(model_load['model_state_dict'])
        train_acc = model_load['train_acc']
        train_loss = model_load['train_loss']
        validation_acc = model_load['validation_acc']
        train_epochs = model_load['epoch']
        config.logger.debug(f"Saved model: Training accuracy: {train_acc}")
        config.logger.debug(f"Saved model: Training loss: {train_loss}")
        config.logger.debug(f"Saved model: Best validation accuracy during training: {validation_acc}")
        config.logger.debug(f"Saved model: Training epochs (at early stopping point): {train_epochs}")
    
    if config.use_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(), lr=config.weights_lr, weight_decay=config.weight_decay
        )
    elif config.use_optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=config.weights_lr, 
                                     weight_decay=config.weight_decay)
    elif config.use_optimizer == "lion":
        optimizer = Lion(
                network.parameters(), lr=config.weights_lr, weight_decay=config.weight_decay
            )
    elif config.use_optimizer == "rmsprop":
        optimizer = RMSpropOptimizerCustom(network.parameters(), default_lr=config.weights_lr)
        if config.nonnegative_params:
            optimizer.nonnegative_hook()
        optimizer.weight_decay_hook(config.weight_decay)
    elif config.use_optimizer == "rmsprop_sliding_window":
        optimizer = RMSpropOptimizerSlidingWindow(network.parameters(), default_lr=config.weights_lr,
                                                learnable_width=config.learnable_width, 
                                                slide_speed=config.slide_speed,
                                                start_col_f = config.start_col_f)
        if config.nonnegative_params:
            optimizer.nonnegative_hook()
        optimizer.weight_decay_hook(config.weight_decay)
    else:
        raise ValueError("bad value")

    epoch = 0
    best_valid_accuracy = 0
    best_epoch = 0
    best_valid_loss = None
    best_valid_accuracy = None
    no_loss_improvement_epochs = 0
    no_accuracy_improvement_epochs = 0
    early_stop = False
    while (epoch < config.epoch_count) and config.run_training and not early_stop:
        config.logger.debug(f"training epoch: {epoch}")
        if config.use_optimizer == "rmsprop_sliding_window":
            if config.experiment_name == "Deterministic learnable window" or config.experiment_name == "Unlearning":
                # Reset learnable window each epoch.
                optimizer.reset_learnable_window()
        loss_accumulator = ValueAccumulator()
        accuracy_accumulator = ValueAccumulator()
        network.train(True)
        batch_count = num_slices // config.batch_size
        for batch_ind in range(batch_count):
            sys.stdout.write(".")
            sys.stdout.flush()
            x, y = train_batcher.get_batch()
            if config.experiment_name == "Unlearning":
                y, batch_is_corrupt = config.unlearning_corrupt_labels_fn(config, y, optimizer, batch_ind, num_classes)
                    
            if config.network_type == "MLP":
                # It takes the batch size as dimension 0.
                y_1hot = F.one_hot(y, num_classes).squeeze().float()
                x = x.t() # put batch size dimension 0.
                y_pred, loss = network(x, y_1hot)
                y_pred = y_pred.t()
            elif config.network_type == "PFCLayer" or config.network_type == "PFCLayer2":
                # It takes the batch size in dimension 1
                y_1hot = F.one_hot(y, num_classes).squeeze().t().float()
                y_pred, loss = network(x, y_1hot)
            else:
                raise ValueError("bad value")         
            loss_accumulator.accumulate(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if config.experiment_name == "Unlearning" and config.skip_optimizer_update_on_corrupt_data:
                if batch_is_corrupt:
                    optimizer.step_without_update()
                else:
                    optimizer.step()
            else:
                optimizer.step()

            if config.nonnegative_params:
                network.clip_weights_nonnegative()
            
            predicted_int = y_pred.argmax(dim=0, keepdim=True)
            batch_size = predicted_int.size(1)
            batch_correct = predicted_int.eq(y).sum().item()
            accuracy_accumulator.accumulate(batch_correct, batch_size)
                
        print("")
        train_mean_loss = loss_accumulator.get_mean()
        config.logger.debug(f"Training loss: {train_mean_loss}")
        train_accuracy = accuracy_accumulator.get_mean()
        config.logger.debug(f"train accuracy: {train_accuracy}")
        
        config.logger.debug('Evaluating model on validation dataset...')
        
        if config.use_optimizer == "rmsprop_sliding_window":
            start_ind = int(optimizer.start_col_f)
            config.slw_start_ind = start_ind
            end_ind = int(optimizer.start_col_f + optimizer.learnable_width)
            config.slw_end_ind = end_ind
            config.logger.debug(f"rmsprop_sliding_window optimizer: Learnable weights window: start_ind: {start_ind} | end_ind: {end_ind}")
        mean_valid_loss, val_accuracy = evaluate_model(config, network, val_batcher, num_classes)
        config.logger.debug(f"Validation loss: {mean_valid_loss}")
        config.logger.debug(f"Validation accuracy: {val_accuracy}")
        
        if False:
            # for debug 
            network.print_stats(do_plots=True)

        if (best_valid_accuracy is None) or (val_accuracy > best_valid_accuracy):
            best_valid_accuracy = val_accuracy
            no_accuracy_improvement_epochs = 0
        else:
            no_accuracy_improvement_epochs += 1
            config.logger.debug(f"No accuracy improvement for {no_accuracy_improvement_epochs} epochs.")

        if config.early_stopping_max_no_accuracy_improvement_epochs is not None:
            if no_accuracy_improvement_epochs >= config.early_stopping_max_no_accuracy_improvement_epochs:
                early_stop = True
                config.logger.debug(f"Stoping trainnig because of no accuracy improvement for {no_accuracy_improvement_epochs} epochs.")

        if (best_valid_loss is None) or (mean_valid_loss < best_valid_loss):
            best_valid_loss = mean_valid_loss
            best_epoch = epoch
            if val_accuracy > best_valid_accuracy:
                best_valid_accuracy = val_accuracy
            config.logger.debug("Saving the model parameters.")
            if config.use_optimizer == "rmsprop_sliding_window" and config.dataset_name == 'split MNIST':
                config.start_col_f = optimizer.start_col_f
            no_loss_improvement_epochs = 0
            if config.dataset_name == 'split MNIST':
                model_save = {
                    'model_state_dict': network.state_dict(),
                    'train_acc': train_accuracy,
                    'train_loss': train_mean_loss,
                    'validation_acc': val_accuracy,
                    'validation_loss': mean_valid_loss,
                    'validation_dataset': dataset_val,
                    'epoch': epoch,
                }
            else:
                model_save = {
                    'model_state_dict': network.state_dict(),
                    'train_acc': train_accuracy,
                    'train_loss': train_mean_loss,
                    'validation_acc': val_accuracy,
                    'validation_loss': mean_valid_loss,
                    'epoch': epoch,
                }
            torch.save(model_save, config.save_model_params_file)
        else:
            no_loss_improvement_epochs += 1
            if no_loss_improvement_epochs == config.early_stopping_max_no_loss_improvement_epochs:
                early_stop = True
                config.logger.debug(f"Early stopping threshold of {config.early_stopping_max_no_loss_improvement_epochs} epochs without andy loss improvement. Stopping training.")
            config.logger.debug(f"Previous better validation loss: {best_valid_loss} and accuracy: {best_valid_accuracy} | occured epoch: {best_epoch} | epochs without improvement: {no_loss_improvement_epochs}")

        epoch += 1

    if config.run_evaluation_on_test:
        config.logger.debug('Evaluating model on test dataset...')
        config.logger.debug("Loading the saved model parameters.")
        model_load = torch.load(config.save_model_params_file)
        config.logger.debug("Loading saved model parameters.")
        network.load_state_dict(model_load['model_state_dict'])
        train_acc = model_load['train_acc']
        train_loss = model_load['train_loss']
        validation_acc = model_load['validation_acc']
        train_epochs = model_load['epoch']
        config.logger.debug(f"Saved model: Training accuracy: {train_acc}")
        config.logger.debug(f"Saved model: Training loss: {train_loss}")
        config.logger.debug(f"Saved model: Best validation accuracy during training: {validation_acc}")
        config.logger.debug(f"Saved model: Training epochs (at early stopping point): {train_epochs}")

        if config.experiment_name == "Unlearning":
            if config.run_training == False:
                config.logger.debug("Removing corrupt knowledge from the model.")
                config.remove_knowledge_fn(network, config.bad_start_col, config.bad_end_col + 1)
                network.print_stats(True) # check plots to verify weights were zeroed.
        
        mean_test_loss, test_accuracy = evaluate_model(config, network, test_batcher, num_classes)
        config.logger.debug(f"Test loss: {mean_test_loss}")
        config.logger.debug(f"Test accuracy: {test_accuracy}")
        config.test_accuracy = test_accuracy

        if config.experiment_name == "OOD visualization":
            # Visualization plots on in distribution and OOD examples.
            config.evaluate_model_ood_vis_fn(config, network, test_batcher, num_classes)

    if config.experiment_name == "Split MNIST":
        return network


def train_and_evaluate_various_classifier():
    """Train and evaluate positive factor networks using PFC blocks and MLP-based classification models.

    This reproduces all of the classification experiments involving the
    fully-connected MLP and PFC-based models on MNIST, Fashion MNIST, and CIFAR10.

    """
    # The following configs contain the default hyperparameters. Each experiment may then
    # overwrite some of the defaults.
    config_MLP1 = AttributeDict(hidden_dim = 300,
                           enable_bias = True,
                           drop_prob = 0,
                           mlp_activation = "gelu",
                           loss_type = "labels_mse", 
    )       

    # 1-layer
    config_PFCBlock1 = AttributeDict(basis_vector_count = 2000,
                               weights_noise_scale = 1e-2,
                           h_noise_scale = 0e-2,
                           nmf_inference_algorithm = "fista",
                           fista_tolerance = None,
                           nmf_inference_iters_no_grad = 0,
                           nmf_is_randomized_iters_no_grad = False,
                           nmf_gradient_iters_with_grad = 100, # 10-100?
                           enable_h_column_normalization = True, # Not needed here, but slightly safer and slower.
                           #loss_type = "labels_mse", 
                           loss_type = "images_labels_mse", # Only works for PFC
                           # Only when "images_labels_mse" is used, specify the relative strength of prediction loss.
                           # When set to 1.0, gives maximum prediction loss and no classification loss.
                           class_label_loss_strength = 0.5,
                           )
    # 2-layer
    config_PFC2Block = AttributeDict(basis_vector_count1 = 300, # 300,
                           basis_vector_count2 = 300,
                            weights_noise_scale = 1e-2,
                           h_noise_scale = 0e-2,
                           nmf_inference_algorithm = "fista",
                           fista_tolerance = None,
                           nmf_inference_iters_no_grad = 0,
                           nmf_is_randomized_iters_no_grad = False,
                           nmf_gradient_iters_with_grad = 100, # 10-100?
                           enable_h_column_normalization = True, # Not needed here, but slightly safer and slower.
                           loss_type = "labels_mse", 
                           #loss_type = "images_labels_mse", # Only works for PFC
                           )
    config_factorized_mlp_base = AttributeDict(device = "cuda",
                                          experiment_name = "",
                                                         weights_lr = 3e-4,
                                                         weight_decay = 1e-4,
                                                         use_optimizer = "rmsprop",
                                                        nonnegative_params = True,
                           batch_size = 200,
                           results_dir = "debug_plots",
                           # Set folder to save plot results for this experiment:
                           plots_folder = "figures",
                           epoch_count = 500,
                           plot_every_epochs = 1,
                           # Stop training if too many epochs elapse without improving validation loss.
                           early_stopping_max_no_loss_improvement_epochs = 20,
                           early_stopping_max_no_accuracy_improvement_epochs = None,
                           run_training = True,
                           run_evaluation_on_test = True,
                           resume_training_from_checkpoint = False,
                           # Fraction of the training set to use for the train split. Rest will be used for validation split.
                           train_split_fraction = 0.85,
                           save_model_params_file = 'saved_models/network_parameters.pth',
                           # MLP settings:
                           config_MLP = config_MLP1,
                           # PFC settings:
                           config_PFCBlock = config_PFCBlock1, # 1 layer
                           config_PFC2Block = config_PFC2Block, # 2 layer
    )
    config_PFC2Block.results_dir = config_factorized_mlp_base.results_dir
    config_PFCBlock1.results_dir = config_factorized_mlp_base.results_dir
    config_MLP1.results_dir = config_factorized_mlp_base.results_dir
    config = config_factorized_mlp_base
    
    # Add logger to the config.
    logger = configure_logger()
    config.logger = logger
    config.config_PFCBlock.logger = logger
    config.config_PFC2Block.logger = logger
    config.config_MLP.logger = logger
        
    ##########################################################################
    # Baseline MLP classifier
    run_mlp_image_classification_experiment = True
    if run_mlp_image_classification_experiment:
        run_baseline_mlp_classifier(config)

    ##########################################################################
    # 1-layer factorized classifier
    run_1_layer_factorized_classification_experiment = True
    if run_1_layer_factorized_classification_experiment:
        run_1_layer_factorized_classifier(config)


    ##########################################################################
    # 2-layer factorized classifier
    run_2_layer_factorized_classification_experiment = True
    if run_2_layer_factorized_classification_experiment:
        run_2_layer_factorized_classifier(config)

    ##########################################################################
    # Continual learning experiments: Split MNIST

    run_split_mnist_experiment = True
    if run_split_mnist_experiment:
        run_split_mnist(config)

    
        
    ##########################################################################
    # Make a deterministic mapping from each training batch to a small region of the model weights

    run_deterministic_learnable_window = True
    if run_deterministic_learnable_window:
        config.experiment_name = "Deterministic learnable window"
        config.weights_lr = 2e-3 # 3e-3 #
        config.weight_decay = 1e-4
        config.use_optimizer = "rmsprop_sliding_window"
        config.network_type = "PFC"
        config.config_PFC.basis_vector_count = 3000
        config.config_PFC.loss_type = "images_labels_mse"
        #config.config_PFC.loss_type = "labels_mse"
        config.dataset_name = 'MNIST'
        #config.dataset_name = 'CIFAR10'
        config.batch_size = 50 # 100
        if config.use_optimizer == "rmsprop_sliding_window":
            # Set initial position of sliding learnable window.
            config.learnable_width=50 # 20
            config.slide_speed=2.5 # 8
            config.start_col_f = 0.0
        run_factorized_or_mlp_classifier(config)

    ##########################################################################
    # Sorted label MNIST

    run_sorted_label_mnist_experiment = True
    if run_sorted_label_mnist_experiment:
        run_sorted_label_mnist(config)



    ##########################################################################
    # Perform unlearning using the deterministic batch to weights region mapping

    run_unlearning_experiment = True
    if run_unlearning_experiment:
        run_unlearning(config)

    
    ##########################################################################
    # OOD visualization: Train on MNIST and evaluate on some OOD examples

    run_ood_mnist_experiment = True
    if run_ood_mnist_experiment:
        run_ood_mnist(config)


def run_baseline_mlp_classifier(config):
    """Train and evaluate an MLP on image classification datasets.

    datasets: MNIST, Fashion MNIST, CIFAR10

    We evaluate with the MLP hidden layer dimension in [300, 2000, 5000]
    """
    datasets = ['MNIST', 'Fashion MNIST', 'CIFAR10']
    mlp_hidden_dims = [300, 2000, 5000]
    num_runs = 3
    config.experiment_name = "MLP classifier"
    config.network_type = "MLP"
    config.epoch_count = 300
    config.nonnegative_params = False
    config.run_training = True
    config.run_evaluation_on_test = True
    config.weights_lr = 1e-4
    config.weight_decay = 1e-4
    config.use_optimizer = "rmsprop"
    config.early_stopping_max_no_loss_improvement_epochs = 20
    config.early_stopping_max_no_accuracy_improvement_epochs = 20
    run_combinations = True
    if run_combinations:
        config.logger.info(f"MLP: datasets: {datasets} | hidden layer dimensions: {mlp_hidden_dims}")
        for dataset in datasets:
            for mlp_hidden_dim in mlp_hidden_dims:
                config.dataset_name = dataset
                config.config_MLP.hidden_dim = mlp_hidden_dim
                sum_accuracy = 0
                for run_ind in range(num_runs):
                    run_factorized_or_mlp_classifier(config)
                    config.logger.info(f"MLP: run: {run_ind} | dataset: {dataset} | hidden_dim: {mlp_hidden_dim} | test accuracy: {config.test_accuracy}")
                    sum_accuracy += config.test_accuracy
                mean_accuracy = sum_accuracy / num_runs
                config.logger.info(f"MLP: dataset: {dataset} | hidden_dim: {mlp_hidden_dim} | mean test accuracy: {mean_accuracy}")



def run_1_layer_factorized_classifier(config):
    """Train and evaluate a positive factor network with 1 PFC block on image classification datasets.

    datasets: MNIST, Fashion MNIST, CIFAR10

    We evaluate with weight template values in [300, 2000, 5000]
    """
    datasets = ['MNIST', 'Fashion MNIST', 'CIFAR10']
    basis_vector_counts = [300, 2000, 5000]
    use_nonneg_params = [True, False]
    num_runs = 1
    config.experiment_name = "1-layer Factorized classifier"
    config.network_type = "PFCLayer"
    config.epoch_count = 300
    config.run_training = True
    config.run_evaluation_on_test = True
    config.weights_lr = 3e-4
    config.weight_decay = 1e-4
    config.use_optimizer = "rmsprop"
    #config.config_PFCBlock.loss_type = "images_labels_mse"
    config.config_PFCBlock.loss_type = "labels_mse"
    config.early_stopping_max_no_loss_improvement_epochs = 20
    config.early_stopping_max_no_accuracy_improvement_epochs = 20
    run_combinations = True
    if run_combinations:
        config.logger.info(f"1-layer factorized: datasets: {datasets} | weight template counts: {basis_vector_counts} | use non-neg params: {use_nonneg_params}")
        for dataset in datasets:
            for basis_vector_count in basis_vector_counts:
                for is_nonneg_param in use_nonneg_params:
                    config.dataset_name = dataset
                    config.config_PFCBlock.basis_vector_count = basis_vector_count
                    config.nonnegative_params = is_nonneg_param
                    sum_accuracy = 0
                    for run_ind in range(num_runs):
                        run_factorized_or_mlp_classifier(config)
                        config.logger.info(f"1-layer factorized: run: {run_ind} | dataset: {dataset} | basis_vector_count: {basis_vector_count} | use non-neg params: {is_nonneg_param} | test accuracy: {config.test_accuracy}")
                        sum_accuracy += config.test_accuracy
                    mean_accuracy = sum_accuracy / num_runs
                    config.logger.info(f"1-layer factorized: dataset: {dataset} | basis_vector_count: {basis_vector_count} | use non-neg params: {is_nonneg_param} | mean test accuracy: {mean_accuracy}")



def run_2_layer_factorized_classifier(config):
    """Train and evaluate a factorized fully-connected residual network on image classification datasets.

    Train and evaluate a residual positive factor network with 2 PFC blocks and 1 skip connection.

    datasets: MNIST, Fashion MNIST, CIFAR10

    We evaluate with weight template values in [300, 2000, 5000]
    """
    datasets = ['MNIST', 'Fashion MNIST', 'CIFAR10']
    basis_vector_counts = [300, 2000, 5000]
    use_nonneg_params = [True, False]
    num_runs = 1
    config.experiment_name = "2-layer Factorized classifier"
    config.network_type = "PFCLayer2"
    config.epoch_count = 300
    config.run_training = True
    config.run_evaluation_on_test = True
    config.weights_lr = 3e-4
    config.weight_decay = 1e-4
    config.use_optimizer = "rmsprop"
    #config.config_PFC2Block.loss_type = "images_labels_mse"
    config.config_PFC2Block.loss_type = "labels_mse"
    config.early_stopping_max_no_loss_improvement_epochs = 20
    config.early_stopping_max_no_accuracy_improvement_epochs = 20
    run_combinations = True
    if run_combinations:
        config.logger.info(f"2-layer factorized: datasets: {datasets} | weight template counts: {basis_vector_counts} | use non-neg params: {use_nonneg_params}")
        config.logger.info(f"Current config:\n{config}")
        config.logger.info(f"Using loss: {config.config_PFC2Block.loss_type}")
        for dataset in datasets:
            for basis_vector_count in basis_vector_counts:
                for is_nonneg_param in use_nonneg_params:
                    config.dataset_name = dataset
                    config.config_PFC2Block.basis_vector_count1 = basis_vector_count
                    config.config_PFC2Block.basis_vector_count2 = basis_vector_count
                    config.nonnegative_params = is_nonneg_param
                    sum_accuracy = 0
                    for run_ind in range(num_runs):
                        run_factorized_or_mlp_classifier(config)
                        config.logger.info(f"2-layer factorized: run: {run_ind} | dataset: {dataset} | basis_vector_count: {basis_vector_count} | use non-neg params: {is_nonneg_param} | test accuracy: {config.test_accuracy}")
                        sum_accuracy += config.test_accuracy
                    mean_accuracy = sum_accuracy / num_runs
                    config.logger.info(f"2-layer factorized: dataset: {dataset} | basis_vector_count: {basis_vector_count} | use non-neg params: {is_nonneg_param} | mean test accuracy: {mean_accuracy}")



def run_split_mnist(config):
    """Continual learning experiments: Split MNIST

    Run the Split MNIST tasks for each combination of optimizer and model below:

    Optimizers: standard RMSprop vs SLW RMSprop.
    Models: MLP vs factorized.
    """
    run_split_MNIST_PFCBlock_1 = False

    if run_split_MNIST_PFCBlock_1:
        config.logger.info("1-layer factorized model on Split MNIST.")
        # Note: Only PFC models are supported by this optimizers
        config.experiment_name = "Split MNIST"
        config.nonnegative_params = True
        config.weights_lr = 3e-4
        config.weight_decay = 1e-4
        config.use_optimizer = "rmsprop_sliding_window"
        config.network_type = "PFC"
        config.config_PFC.basis_vector_count = 2000
        config.config_PFC.loss_type = "images_labels_mse"
        config.dataset_name = 'split MNIST'
        config.early_stopping_max_no_loss_improvement_epochs = 5
        config.early_stopping_max_no_accuracy_improvement_epochs = 5
        # Each split corresponds to a binary classification task on two digits, so 5 splits in total.
        splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        # For evaluation over all seen classes
        all_seen_classes = []

        if config.use_optimizer == "rmsprop_sliding_window":
            # Set initial position of sliding learnable window.
            config.learnable_width=15
            config.slide_speed=0.25
            config.start_col_f = 0.0

        for split_index, split in enumerate(splits):
            config.logger.debug(f"Training on split: {split}")
            config.split = split
            all_seen_classes.extend(split)
            config.all_seen_classes = all_seen_classes
            config.logger.debug(f"all_seen_classes: {all_seen_classes}")
            config.split_index = split_index
            if split_index == 0:
                # Train from scratch on first split.
                config.resume_training_from_checkpoint = False
            else:
                config.resume_training_from_checkpoint = True
            
            network = run_factorized_or_mlp_classifier(config)
            Wx = network.Wx # input reconstruction weights
            Wy = network.Wy # output classification weights
            plot_image_count = 25
            left_padding = 25 # Optional amount of padding to the left of the learnable window left border. Can be 0.
            # Get the weights to the left of the current learnable window's left border.
            start_ind = config.slw_start_ind - plot_image_count - left_padding
            assert start_ind >= 0
            end_ind = start_ind + plot_image_count
            Wx_just_learned = Wx[:, start_ind:end_ind]
            plot_rows_as_images(Wx_just_learned.t(), 
                            file_name=f"{config.plots_folder}/split_mnist/slw_optimizer/weights_Wx_just_learned_split_index_{split_index}.png", 
                            img_height=28, img_width=28, plot_image_count=plot_image_count)
            
            plot_image_matrix(Wx, file_name=f"{config.plots_folder}/split_mnist/slw_optimizer/weights_Wx_split_index_{split_index}.png", 
                                title=f"Reconstruction weights after learning split {split_index + 1}/5",
                                xlabel="column index",
                                ylabel="feature index")
            plot_image_matrix(Wy, file_name=f"{config.plots_folder}/split_mnist/slw_optimizer/weights_Wy_split_index_{split_index}.png", 
                                title=f"Classification weights after learning split {split_index + 1}/5",
                                xlabel="column index",
                                ylabel="feature index")
            

        config.logger.info(f"Factorized 1-layer: Test accuracy after training on all splits: {config.test_accuracy}")
        # Test accuracy: 0.9373
        # rmsprop_sliding_window optimizer: Learnable weights window: start_ind: 1342 | end_ind: 1357

    run_split_MNIST_PFC_1Block_standard_optimizer = False
    if run_split_MNIST_PFC_1Block_standard_optimizer:
        config.logger.info("1-layer factorized model on Split MNIST. Using standard optimzier.")
        # Note: Only PFC models are supported by this optimizers
        config.experiment_name = "Split MNIST"
        config.nonnegative_params = True
        config.weights_lr = 1e-5 # 1e-5
        config.weight_decay = 1e-4 # 1e-4
        config.use_optimizer = "rmsprop"
        config.network_type = "PFC"
        config.config_PFC.basis_vector_count = 1357 # Same in-use basis vectors as factorized network with sliding window optimizer
        config.config_PFC.loss_type = "images_labels_mse"
        config.dataset_name = 'split MNIST'
        config.early_stopping_max_no_loss_improvement_epochs = 5
        config.early_stopping_max_no_accuracy_improvement_epochs = 5
        # Each split corresponds to a binary classification task on two digits, so 5 splits in total.
        splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        # For evaluation over all seen classes
        all_seen_classes = []

        if config.use_optimizer == "rmsprop_sliding_window":
            # Set initial position of sliding learnable window.
            config.learnable_width=15
            config.slide_speed=0.25
            config.start_col_f = 0.0

        for split_index, split in enumerate(splits):
            config.logger.debug(f"Training on split: {split}")
            config.split = split
            all_seen_classes.extend(split)
            config.all_seen_classes = all_seen_classes
            config.logger.debug(f"all_seen_classes: {all_seen_classes}")
            config.split_index = split_index
            if split_index == 0:
                # Train from scratch on first split.
                config.resume_training_from_checkpoint = False
            else:
                config.resume_training_from_checkpoint = True
            
            network = run_factorized_or_mlp_classifier(config)
            # We plot the same reconstruction weights plots after each split. This allows us to visualize the
            # forgetting, since we can see previously learned digit patterns start to gradually fade away as
            # more splits are learned.
            Wx = network.Wx 
            plot_rows_as_images(Wx.t(), 
                            file_name=f"{config.plots_folder}/split_mnist/weights_Wx_split_index_{split_index}.png", 
                            img_height=28, img_width=28, plot_image_count=100)
            

        config.logger.info(f"Factorized 1-layer: Using standard optimizer: Test accuracy after training on all splits: {config.test_accuracy}")
        # Test accuracy: 0.6479
        # Test accuracy: 0.6896
        # Test accuracy: 0.6406
        # Test accuracy: 0.6791
        # Test accuracy: 0.6965

    run_split_MNIST_PFC_2Block = False

    if run_split_MNIST_PFC_2Block:
        config.logger.info("2-layer factorized model on Split MNIST.")
        config.nonnegative_params = True
        config.experiment_name = "Split MNIST"
        # Note: Only PFC models are supported by this optimizers
        config.weights_lr = 3e-4
        config.use_optimizer = "rmsprop_sliding_window"
        config.network_type = "SF2Layer"
        config.config_PFC2Block.basis_vector_count1 = 2000
        config.config_PFC2Block.basis_vector_count2 = 2000
        config.config_PFC2Block.loss_type = "images_labels_mse"
        config.dataset_name = 'split MNIST'
        config.early_stopping_max_no_loss_improvement_epochs = 5
        config.early_stopping_max_no_accuracy_improvement_epochs = 5
        # Each split corresponds to a binary classification task on two digits, so 5 splits in total.
        splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        # For evaluation over all seen classes
        all_seen_classes = []

        if config.use_optimizer == "rmsprop_sliding_window":
            # Set initial position of sliding learnable window.
            config.learnable_width=15
            config.slide_speed=0.25
            config.start_col_f = 0.0

        for split_index, split in enumerate(splits):
            config.logger.debug(f"Training on split: {split}")
            config.split = split
            all_seen_classes.extend(split)
            config.all_seen_classes = all_seen_classes
            config.logger.debug(f"all_seen_classes: {all_seen_classes}")
            config.split_index = split_index
            if split_index == 0:
                # Train from scratch on first split.
                config.resume_training_from_checkpoint = False
            else:
                config.resume_training_from_checkpoint = True
            
            run_factorized_or_mlp_classifier(config)

        config.logger.info(f"Factorized 2-layer:Test accuracy after training on all splits: {config.test_accuracy}")

    run_split_MNIST_mlp = False
    if run_split_MNIST_mlp:
        config.logger.info("MLP on Split MNIST")
        config.experiment_name = "Split MNIST"
        config.nonnegative_params = False
        config.weights_lr = 2e-6 # Note: MLP requires very low lr here.
        config.weight_decay = 1e-4
        config.use_optimizer = "rmsprop"
        config.network_type = "MLP"
        config.dataset_name = 'split MNIST'
        config.config_MLP.hidden_dim = 1357 # Same size as factorized network
        config.early_stopping_max_no_loss_improvement_epochs = 5
        config.early_stopping_max_no_accuracy_improvement_epochs = 5
        # Each split corresponds to a binary classification task on two digits, so 5 splits in total.
        splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        # For evaluation over all seen classes
        all_seen_classes = []

        for split_index, split in enumerate(splits):
            #print(f"Training on split: {split}")
            config.logger.debug(f"Training on split: {split}")
            config.split = split
            all_seen_classes.extend(split)
            config.all_seen_classes = all_seen_classes
            #print(f"all_seen_classes: {all_seen_classes}")
            config.logger.debug(f"all_seen_classes: {all_seen_classes}")
            config.split_index = split_index
            if split_index == 0:
                # Train from scratch on first split.
                config.resume_training_from_checkpoint = False
            else:
                config.resume_training_from_checkpoint = True
            
            run_factorized_or_mlp_classifier(config)

        #print(f"MLP: Test accuracy after training on all splits: {config.test_accuracy}")
        config.logger.info(f"MLP: Test accuracy after training on all splits: {config.test_accuracy}")
        # Test accuracy: 0.4011


def run_sorted_label_mnist(config):
    """Sorted label MNIST

    Demonstration of non-iid training.
    Train on MNIST where the digits are sorted ascending within each epoch with
    deterministic ordering. So, first all the 0's are presented, followed by all
    the 1's, then all the 2's and so on.
    The factorized network uses "rmsprop_sliding_window" optimizer (also train factorized
    with same optimizer as MLP for comparison).

    """
    run_sub_experiment_factorized_and_shuffled_sliding_window = True
    run_sub_experiment_factorized_and_shuffled_regular_optimizer = True
    run_sub_experiment_factorized_and_sorted_order_sliding_window = True
    run_sub_experiment_factorized_and_sorted_order_regular_optimizer = True
    run_sub_experiment_mlp_and_sorted_order_regular_optimizer = True
    run_sub_experiment_mlp_and_shuffled_regular_optimizer = True

    config.experiment_name = "Deterministic learnable window"
    config.epoch_count = 100 # 
    config.early_stopping_max_no_loss_improvement_epochs = 5
    config.early_stopping_max_no_accuracy_improvement_epochs = 5
    config.weights_lr = 2e-3 # 
    config.weight_decay = 1e-4
    config.use_optimizer = "rmsprop_sliding_window"
    config.network_type = "PFC"
    config.config_PFC.basis_vector_count = 3000
    config.config_PFC.loss_type = "images_labels_mse"
    config.dataset_name = 'ordered MNIST'
    config.ordered_shuffle = True
    config.batch_size = 50 # 100
        
    # Set initial position of sliding learnable window.
    config.learnable_width=50 #
    config.slide_speed=2.5 #
    config.start_col_f = 0.0
    factorized_and_shuffled_sliding_window_accuracy = ""
    if run_sub_experiment_factorized_and_shuffled_sliding_window:
        print("Running factorized model with examples in fixed shuffled ordering. Using sliding window optimizer.") 
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.9602
        factorized_and_shuffled_sliding_window_accuracy = config.test_accuracy
        print("Finished factorized model with examples in fixed shuffled ordering. Using sliding window optimizer.")
    config.use_optimizer = "rmsprop"
    config.config_PFC.basis_vector_count = 2600
        
    factorized_and_shuffled_regular_optimizer = ""
    if run_sub_experiment_factorized_and_shuffled_regular_optimizer:
        print("Running factorized model with examples in fixed shuffled ordering. Using RMSprop optimizer.") 
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.9788
        factorized_and_shuffled_regular_optimizer = config.test_accuracy
        print("Finished factorized model with examples in fixed shuffled ordering. Using RMSprop optimizer.")
    config.config_PFC.basis_vector_count = 3000
    config.ordered_shuffle = False
    config.use_optimizer = "rmsprop_sliding_window"
    factorized_and_sorted_order_sliding_window = ""
    if run_sub_experiment_factorized_and_sorted_order_sliding_window:
        print("Running factorized model with examples in sorted label ordering. Using sliding window optimizer.") 
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.9624
        factorized_and_sorted_order_sliding_window = config.test_accuracy
        print("Finished factorized model with examples in sorted label ordering. Using sliding window optimizer.")
    config.use_optimizer = "rmsprop"
    config.config_PFC.basis_vector_count = 2600
    factorized_and_sorted_order_regular_optimizer = ""
    if run_sub_experiment_factorized_and_sorted_order_regular_optimizer:
        print("Running factorized model with examples in sorted label ordering. Using RMSprop optimizer.") 
        config.weights_lr = 5e-6
        config.weight_decay = 0e-5
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.8819
        factorized_and_sorted_order_regular_optimizer = config.test_accuracy
        print("Finished factorized model with examples in sorted label ordering. Using RMSprop optimizer.")
    # Switch to MLP
    config.network_type = "MLP"
    config.nonnegative_params = False
    config.weights_lr = 5e-7 # 5e-7
    config.weight_decay = 0e-5
    config.config_MLP.hidden_dim = 2600
    config.ordered_shuffle = False
    config.epoch_count = 200 # MLP converges slow with low lr, so give it more epochs.
    mlp_and_sorted_order_regular_optimizer = ""
    if run_sub_experiment_mlp_and_sorted_order_regular_optimizer:
        print("Running MLP model with examples in sorted label ordering. Using RMSprop optimizer.") 
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.8492
        mlp_and_sorted_order_regular_optimizer = config.test_accuracy
        print("Finished MLP model with examples in sorted label ordering. Using RMSprop optimizer.") 

    config.ordered_shuffle = True
    config.weights_lr = 1e-4 
    config.weight_decay = 1e-4
    mlp_and_shuffled_regular_optimizer = ""
    if run_sub_experiment_mlp_and_shuffled_regular_optimizer:
        print("Running MLP model with examples in fixed shuffled ordering. Using RMSprop optimizer.") 
        run_factorized_or_mlp_classifier(config) # Test accuracy: 0.9791
        mlp_and_shuffled_regular_optimizer = config.test_accuracy
        print("Finished MLP model with examples in fixed shuffled ordering. Using RMSprop optimizer.")

    print("")
    print("Results summary:")
    print("Factorized model results:")
    print(f"Factorized model with examples in fixed shuffled ordering. Using sliding window optimizer. Test accuracy: {factorized_and_shuffled_sliding_window_accuracy}")
    print(f"Factorized model with examples in fixed shuffled ordering. Using RMSprop optimizer. Test accuracy: {factorized_and_shuffled_regular_optimizer}")
    print(f"Factorized model with examples in sorted label ordering. Using sliding window optimizer. Test accuracy: {factorized_and_sorted_order_sliding_window}")
    print(f"Factorized model with examples in sorted label ordering. Using RMSprop optimizer. Test accuracy: {factorized_and_sorted_order_regular_optimizer}")
    print("MLP model results:")
    print(f"MLP model with examples in sorted label ordering. Using RMSprop optimizer. Test accuracy: {mlp_and_sorted_order_regular_optimizer}")
    print(f"MLP model with examples in fixed shuffled ordering. Using RMSprop optimizer. Test accuracy: {mlp_and_shuffled_regular_optimizer}")
    #Results summary:
    #Factorized model results:
    #Factorized model with examples in fixed shuffled ordering. Using sliding window optimizer. Test accuracy: 0.9589
    #Factorized model with examples in fixed shuffled ordering. Using RMSprop optimizer. Test accuracy: 0.9799
    #Factorized model with examples in sorted label ordering. Using sliding window optimizer. Test accuracy: 0.9541
    #Factorized model with examples in sorted label ordering. Using RMSprop optimizer. Test accuracy: 0.8846
    #MLP model results:
    #MLP model with examples in sorted label ordering. Using RMSprop optimizer. Test accuracy: 0.8527
    #MLP model with examples in fixed shuffled ordering. Using RMSprop optimizer. Test accuracy: 0.9791


def run_unlearning(config):
    """Run unlearning experiment.

    Perform unlearning using the deterministic batch to weights region mapping

    We will corrupt the labels of a range of training batches (in deterministic order) in
    each epoch. This will of course reduce the accuracy on the validation/test sets.
    We can then fix the problem in the trained model by removing the portion of the weights associated with
    these bad examples, restoring accuracy.

    """
    config.experiment_name = "Unlearning"
    config.weights_lr = 1e-3
    config.weight_decay = 1e-4
    config.use_optimizer = "rmsprop_sliding_window"
    config.nonnegative_params = True
    config.network_type = "PFC"
    # Be sure to choose it large enough to leave a little padding of unused weights.
    # It does not hurt accuracy even to leave a large amount of unused weights.
    config.config_PFC.basis_vector_count = 3000 
    config.config_PFC.loss_type = "images_labels_mse"
    config.dataset_name = 'MNIST'
    #config.dataset_name = 'CIFAR10'
    config.batch_size = 50
    config.starting_corrupt_batch_index = 500 # roughly in the middle
    config.corrupt_batch_count = 300 # 300
    # Empirically observed as the maximum number of epochs required when using corrupted data
    config.epoch_count = 8
    config.early_stopping_max_no_loss_improvement_epochs = 5
    if config.use_optimizer == "rmsprop_sliding_window":
        # Set initial position of sliding learnable window.
        config.learnable_width=50
        config.slide_speed=2.5
        config.start_col_f = 0.0

    config.skip_optimizer_update_on_corrupt_data = False # Include the range of bad batches when training

    def corrupt_some_labels(config, y, optimizer, batch_ind, num_classes):
        """Corrupt a range of labels for unlearning and identify start/end of bad data in weights.

        Args:
            config
            y (tensor): class labels for batch
            optimizer
        """
        ending_corrupt_batch_index = config.starting_corrupt_batch_index + config.corrupt_batch_count
        if (batch_ind >= config.starting_corrupt_batch_index) and (batch_ind < ending_corrupt_batch_index):
            # corrupt labels in y.
            # y contains the integer class label, so the following will make sure it is wrong.
            y = (y + 1) % num_classes
            batch_is_corrupt = True
        else:
            batch_is_corrupt = False

        if batch_ind == config.starting_corrupt_batch_index:
            start_col, _ = optimizer.get_learnable_window_borders()
            # Save location of the last bad column of the weights
            config.bad_start_col = start_col

        if batch_ind == ending_corrupt_batch_index:
            _, end_col = optimizer.get_learnable_window_borders()
            # Save location of the last bad column of the weights
            config.bad_end_col = end_col
        return y, batch_is_corrupt
    
    config.unlearning_corrupt_labels_fn = corrupt_some_labels
    # Train the model on data that includes a region of bad (corrupt) training batches for
    #  batches indices in [config.starting_corrupt_batch_index, config.starting_corrupt_batch_index + config.corrupt_batch_count]
    #  and then evaluate on test set:
    run_factorized_or_mlp_classifier(config)
    test_accuracy_on_good_and_bad_data = config.test_accuracy
    
    # Evaluate on test set with the corrupt data removed.
    config.run_training = False

    # function to remove knowledge from the model:
    def remove_knowledge(network, start_col, end_col):
        # Plot the weights before unlearning:
        Wx = network.Wx 
        plot_image_matrix(Wx, file_name=f"{config.plots_folder}/unlearning/weights_before_unlearning_Wx.png", 
                                title="Reconstruction weights before unlearning",
                                xlabel="column index",
                                ylabel="feature index")
        Wy = network.Wy 
        plot_image_matrix(Wy, file_name=f"{config.plots_folder}/unlearning/weights_before_unlearning_Wy.png", 
                                title="Classification weights before unlearning",
                                xlabel="column index",
                                ylabel="feature index")

        # Perform the unlearning operation here:
        with torch.no_grad():
            params = network.parameters()
            for param in params:
                assert param.ndim == 2, "Param has bad size. Should be 2-dim."
                param[:, start_col:end_col] *= 0
        
        # Plot the unlearned weights:
        Wx = network.Wx 
        plot_image_matrix(Wx, file_name=f"{config.plots_folder}/unlearning/weights_unlearned_Wx.png", 
                                title="Reconstruction weights after unlearning",
                                xlabel="column index",
                                ylabel="feature index")
        Wy = network.Wy 
        plot_image_matrix(Wy, file_name=f"{config.plots_folder}/unlearning/weights_unlearned_Wy.png", 
                                title="Classification weights after unlearning",
                                xlabel="column index",
                                ylabel="feature index")

    config.remove_knowledge_fn = remove_knowledge    
    run_factorized_or_mlp_classifier(config)
    test_accuracy_after_unlearning = config.test_accuracy
    print("Retraining model from scratch on only the good subset. Leaving out the corrupt batches")
    config.run_training = True
    config.skip_optimizer_update_on_corrupt_data = True # Only train on the good subset of the dataset
    run_factorized_or_mlp_classifier(config)
    test_accuracy_upper_bound_for_unlearning = config.test_accuracy
    print("")
    print("Results summary:")
    print(f"Test accuracy after being trained on both good data and a region of bad/corrupt batches: {test_accuracy_on_good_and_bad_data}")
    print(f"Location of the bad/corrupt knowledge in weights Wx and Wy: starting column index: {config.bad_start_col} | ending column index: {config.bad_end_col}")
    print(f"Test accuracy after unlearning: {test_accuracy_after_unlearning}")
    print(f"Test accuracy when trained only on the good batches (accuracy upper bound after unlearning): {test_accuracy_upper_bound_for_unlearning}")

    

def run_ood_mnist(config):
    """Run OOD visualization experiment: Train on MNIST and evaluate on some OOD examples

    Here, we train a factorized model on MNIST and then evaluate the model on both some MNIST (in domain)
    and some Fashion MNIST (OOD) examples. 
    
    """
    @torch.no_grad()
    def evaluate_model_ood_vis_fn(config, network, mnist_data_batcher, num_classes):
        
        assert config.network_type == "PFC" or config.network_type == "SF2Layer"
        # Load the Fashion MNIST test set which we use as the source of OOD examples.
        dataset_test_ood = datasets.FashionMNIST(datasets_root, train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
        
        test_batcher_ood = classification_dataset_to_batcher(dataset_test_ood, config.batch_size, config.device, False)
        network.train(False)
        
        # Evaluate on in-distribution examples (1 batch only)
        x, y = mnist_data_batcher.get_batch()
        x_dim, batch_size = x.size()
        assert x_dim == 28*28
        img_height = 28
        img_width = 28
        y_1hot = F.one_hot(y, num_classes).squeeze().t().float()
        # y_1hot shape: (num_classes, batch_size)
        # x shape: (x_dim, batch_size)
        # y_pred shape: (num_classes, batch_size)
        y_pred, loss = network(x, y_1hot)

        # Plot in-distribution (MNIST) inputs as images:
        x_rows_as_images = x.t()
        plot_rows_as_images(x_rows_as_images, 
                            file_name=f"{config.plots_folder}/ood_visualization/input_in_distribution_images.png", 
                            img_height=img_height, img_width=img_width, plot_image_count=None)
        
        # Plot reconstructed input images
        h_tran = network.h_tran
        Wx = network.Wx
        reconstructed_input = torch.einsum("ij,kj->ki", Wx, h_tran)
        plot_rows_as_images(reconstructed_input, 
                            file_name=f"{config.plots_folder}/ood_visualization/input_in_distribution_reconstructed_lambda_{config.config_PFC.class_label_loss_strength}.png", 
                            img_height=img_height, img_width=img_width, plot_image_count=None)

        # Plot weights as images
        plot_rows_as_images(Wx.t(), 
                            file_name=f"{config.plots_folder}/ood_visualization/weights_Wx_as_images_lambda_{config.config_PFC.class_label_loss_strength}.png", 
                            img_height=img_height, img_width=img_width, plot_image_count=None)
        
        # Evaluate on OOD examples (1 batch only)
        x, y = test_batcher_ood.get_batch()
        x_dim, batch_size = x.size()
        assert x_dim == 28*28
        y_1hot = F.one_hot(y, num_classes).squeeze().t().float()
        y_pred, loss = network(x, y_1hot)

        # Plot OOD (Fashion MNIST) inputs as images:
        x_rows_as_images = x.t()
        plot_rows_as_images(x_rows_as_images, 
                            file_name=f"{config.plots_folder}/ood_visualization/input_ood_images.png", 
                            img_height=img_height, img_width=img_width, plot_image_count=None)
        
        h_tran = network.h_tran
        Wx = network.Wx
        reconstructed_input = torch.einsum("ij,kj->ki", Wx, h_tran)
        plot_rows_as_images(reconstructed_input, 
                            file_name=f"{config.plots_folder}/ood_visualization/input_ood_reconstructed.png", 
                            img_height=img_height, img_width=img_width, plot_image_count=None)

        print("")
    config.experiment_name = "OOD visualization"
    config.dataset_name = 'MNIST'
    config.evaluate_model_ood_vis_fn = evaluate_model_ood_vis_fn
    config.epoch_count = 5 # 5
    config.network_type = "PFC"
    config.nonnegative_params = True
    config.weights_lr = 1e-3 # 1e-3
    config.weight_decay = 1e-4
    config.batch_size = 50
    config.config_PFC.basis_vector_count = 100
    config.config_PFC.loss_type = "images_labels_mse"
    config.config_PFC.nmf_inference_iters_no_grad = 0 # 200
    config.config_PFC.nmf_gradient_iters_with_grad = 100 # 100
    # Now train with different relative strengths of the two loss terms:
    # image reconstruction quality vs class prediction quality.
    results = []
    for config.config_PFC.class_label_loss_strength in [1.0, 0.9, 0.1, 0.0, 0.5]:
        run_factorized_or_mlp_classifier(config)
        results.append((config.config_PFC.class_label_loss_strength, config.test_accuracy))
        print(f"(Using class_label_loss_strength: {config.config_PFC.class_label_loss_strength}) Accuracy on in-distribution test set (MNIST): {config.test_accuracy}")
        
    # Summarize results:
    print("Results summary:")
    for item in results:
        print(f"Using class_label_loss_strength: {item[0]} | Accuracy on in-distribution test set (MNIST): {item[1]}")
        
    # Reaches  0.9497 accuracy on MNIST test set with class_label_loss_strength = 0.5.




@torch.no_grad()
def train_and_evaluate_learning_repeated_sequence():
    """
    Train a non-negative factorized RNN using standard NMF updates on a deterministic sequence memorization task.

    Uses SGD updates for both W and H factor matrices with auto FISTA lr.

    Note: Since the network's task is to memorize a repeating deterministic sequence, there is no generalization
    required. So, the validation sequence is the same as the training sequence.
   
    """
    # Use standard NMF left and right updates. Backprop is not used.
    config_VanillaFactorizedRNN = AttributeDict(use_model = "VanillaFactorizedRNN",
        basis_vector_count = 100,
        # Number of iterations to allow the inference algorithm to hopefully converge
        nmf_inference_iters = 100,
        evaluation_nmf_inference_iteration_count = 100,
        weights_noise_scale = 2e-2,
        weight_decay_H = 0e-4,
        weight_decay_W = 0e-4,
        sparsity_L1_H = 0e-4,
        enforce_nonneg_params = True,
    )

    # Uses standard NMF updates for W and H.
    config_standard_NMF_RNN = AttributeDict(
        device = "cpu",
        dataset = "sample_new_v1",
        batch_size = 100,
        train_seq_len = 200,
        validation_seq_len = 200,
        validation_seed_len=15,
        generate_token_count = 50,
        seed_seq_len = 175,
        train_iterations = 500,
        use_optimizer="internal",
        learning_rate=None,
        weight_decay = 0,
        
        # Print training progress every this many iterations.
        show_progress_every = 100,
        run_evaluation_every = 100,
        sample_every = 100,
        model_config = config_VanillaFactorizedRNN,
        results_dir = "figures/deterministic_nmf_rnn"
    )
    config_VanillaFactorizedRNN.results_dir = config_standard_NMF_RNN.results_dir
    config_VanillaFactorizedRNN.device = config_standard_NMF_RNN.device
    config = config_standard_NMF_RNN
    debugging = True
    if debugging:
        logger = configure_logger(log_results_file='experimental_results.log', log_debug_file='debug.log')
    else:
        logger = configure_logger()
    config.logger = logger
    config_VanillaFactorizedRNN.logger = logger

    config.logger.info("Standard NMF RNN on deterministic sequence memorization.")

    if config.dataset == "sample_new_v1":
        # Use the Kumozu repeating sequence.
        assert config.validation_seed_len < config.validation_seq_len
        # Sample from transition FSM to create some synthetic training data:
        samples_train = sample_new_v1(config.train_seq_len + 1)
        if False:
            # debug: Try adding a small a mount of noise to check robustness
            # (it can still learn the model as long as the noise is not too high)
            additive_noise = torch.rand_like(samples_train)*0.05
            samples_train = samples_train + additive_noise
        config.model_config.x_dim = 4
        config.model_config.y_dim = 4
        # create validation sequence  (it's the same as the training sequence in this case)
        samples_valid = sample_new_v1(config.validation_seq_len + 1)
        x_train = samples_train[:, :-1]
        assert x_train.size() == (config.model_config.x_dim, config.train_seq_len)
        y_train_targets = samples_train[:, 1:]
        plot_image_matrix(
            x_train, f"{config.results_dir}/training_sequence.png", title="Training sequence",
            xlabel="Time slice index",
            ylabel="feature index"
        )
    
        x_valid = samples_valid[:, :-1]
        assert x_valid.size() == (config.model_config.x_dim, config.validation_seq_len)
        y_valid_targets = samples_valid[:, 1:]
    else:
        raise ValueError("Bad dataset name.")

    network = FactorizedRNNWithoutBackprop(config.model_config)
            
    h_next = None
    for iter in range(config.train_iterations):
        sys.stdout.write(".")
        sys.stdout.flush()
        
        x_train = x_train.to(config.device)
        y_train_targets = y_train_targets.to(config.device)
        y_pred, h_next, loss = network.forward(x_train, y_train_targets, h_prev_init=h_next)
        if iter % 20 == 0:
            config.logger.debug(f"iteration: {iter} | training loss: {loss.item()}")
        
        network.update_weights()
        if iter % config.show_progress_every == 0:
            config.logger.debug("")
            config.logger.debug(f"iteration: {iter} | training loss: {loss.item()}")
            network.print_stats(description="train")
        if iter % config.run_evaluation_every == 0:
            config.logger.debug("Running validation...")
            x_valid = x_valid.to(config.device)
            y_valid_targets = y_valid_targets.to(config.device)
            _, _, loss = network.forward(x_valid, y_valid_targets)
            config.logger.debug(f"validation loss: {loss.item()}")
            
        if iter % config.sample_every == 0:
            config.logger.debug("Sampling from seed sequence...")
            # Split validation samples into two parts: the seed (which is supplied to the model) and the part to be predicted.
            x_valid_seed = samples_valid[:, : config.validation_seed_len]
            if False:
                # debug: Try adding a small a mount of noise to check robustness
                # (it can still generate the correct sequence as long as the noise is not too high)
                additive_noise = torch.rand_like(x_valid_seed)*0.05
                x_valid_seed = x_valid_seed + additive_noise
            plot_image_matrix(
                x_valid_seed, f"{config.results_dir}/seed_seq.png", title="Seed for generated sequence"
            )
            x_valid_seed = x_valid_seed.to(config.device)
            gen_seq = network.generate_from_seed(x_valid_seed, config.generate_token_count)
            network.print_stats(description="generation")
            plot_image_matrix(
                gen_seq, f"{config.results_dir}/generated_seq.png", title="Generated sequence including seed"
            )


def cleanup_debug_plots_folder():
    """Remove any plots (.png files) in the debug plots folder.
    """
    folder_path = "./debug_plots/"
    extension = "*.png"
    files_to_remove = glob.glob(os.path.join(folder_path, extension))

    for file_path in files_to_remove:
        os.remove(file_path)
    


if __name__ == "__main__":
    cleanup_debug_plots_folder()
    # torch.set_num_threads(16)
    # torch.set_flush_denormal(True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # choose experiment to run:
    run_experiment = 'run_sequential_mnist_rnn_experiments'

    if run_experiment == 'train_and_evaluate_various_classifier':
        # Run classifier experiments on MLP-like models. (Use for paper)
        train_and_evaluate_various_classifier()
    elif run_experiment == 'train_and_evaluate_learning_repeated_sequence':
        # RNN using standard NMF updates on deterministic sequence memorization task.
        train_and_evaluate_learning_repeated_sequence()
    elif run_experiment == 'train_and_evaluate_copy_task_factorized_rnn':
        # Factorized RNN using standard NMF updates on the copy task.
        train_and_evaluate_copy_task_factorized_rnn()
    elif run_experiment == 'train_and_evaluate_copy_task_vanilla_rnn':
        # Conventional RNN with and without BPTT on the copy task.
        train_and_evaluate_copy_task_vanilla_rnn()
    elif run_experiment == 'run_sequential_mnist_rnn_experiments':
        # Run experiments on factorized and conventional RNNs on the Sequential MNIST task (uses backprop).
        run_sequential_mnist_rnn_experiments()
    elif run_experiment == 'sequential_mnist_factorized_rnn_conventional_nmf':
        # Run experiment on factorized RNNs on the Sequential MNIST task using conventional NMF W and H updates (no backprop).
        sequential_mnist_factorized_rnn_conventional_nmf()
    else:
        raise ValueError("Bad experiment name.")
