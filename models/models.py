
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
import os



class MLP(nn.Module):
    """A basic MLP.


    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1    = nn.Linear(config.input_dim, 
                                config.hidden_dim, 
                                bias=config.enable_bias)
        self.fc2  = nn.Linear(config.hidden_dim, 
                                 config.output_dim, 
                                 bias=config.enable_bias)
        self.dropout = nn.Dropout(config.drop_prob)
        if config.mlp_activation == "sparse2":
            # (experimental)
            self.sparse_layer = VogelSparseThreshold(keep_frac = 0.05, lr = 1e-3)

    def clip_weights_nonnegative(self):
        """Not recommended. It does not work well in this architecture.
        """
        print("Warning: MLP does not perform well with non-negative weights")
        min_val = 1e-5
        with torch.no_grad():
            self.fc1.weight.data = torch.clamp(self.fc1.weight.data, min=min_val)
            self.fc1.bias.data = torch.clamp(self.fc1.bias.data, min=min_val)
            self.fc2.weight.data = torch.clamp(self.fc2.weight.data, min=min_val)
            self.fc2.bias.data = torch.clamp(self.fc2.bias.data, min=min_val)


    def forward(self, x, y_targets = None):
        """Forward pass.

        Given input `x` to the MLP, compute the output `y_pred`.
        If `y_targets` is `None`, return only `y_pred`. Otherwise, compute the loss (specified
        in the `config` that was supplied to __init__()) and return a tuple containing
        (`y_pred`, loss).

        Args:
            x (Tensor): float tensor of shape (batch_size, x_dim) containing the input features having
                dimension `x_dim`.
            y_targets (Tensor): float tensor of shape (batch_size, y_dim) containing the target features.

        Returns:
            (y_pred, loss) where
                y_pred (Tensor): float tensor of shape (batch_size, y_dim) containing the
                    class label predictions. (the predicted label will be the maximum activated
                    label index).
                loss (Tensor): float tensor of shape (1,) containing the loss. The type of
                    loss to use is specified in the `config` instance that was passed to the
                    constructor.
        
        """
        x = self.fc1(x)
        if self.config.mlp_activation == "gelu":
            x = F.gelu(x)
        elif self.config.mlp_activation == "softmax":
            x = F.softmax(x)
        elif self.config.mlp_activation == "relu":
            x = F.relu(x)
        elif self.config.mlp_activation == "relu-norm":
            # (experimental)
            x = F.relu(x)
            x = F.normalize(x, p=1, dim=1)
        elif self.config.mlp_activation == "sparse2":
            # (experimental)
            x = self.sparse_layer(x)
            x = F.gelu(x)
            #x = F.normalize(x, p=1, dim=1)
        else:
            raise ValueError("oops")
        x = self.fc2(x)
        y_pred = self.dropout(x)
        if y_targets is None:
            return y_pred
        else:
            if self.config.loss_type == "labels_mse":
                # Compute loss only on the label predictions
                loss = torch.nn.functional.mse_loss(y_pred, y_targets, reduction='mean')
            else:
                raise ValueError("bad value")
            return y_pred, loss
    

    @torch.no_grad()
    def print_stats(self, do_plots = False):
        print("key weights: min value: {}; max value: {}".format(self.fc1.weight.min(), 
                                                                 self.fc1.weight.max()))
        print("value weights: min value: {}; max value: {}".format(self.fc2.weight.min(), 
                                                                   self.fc2.weight.max()))


class VanillaRNN(nn.Module):
    """A basic vanilla RNN.

    This implements the vanilla RNN for 1 time slice.

    Inputs:
        h_prev: The internal state vector from the previous time slice.
        x_t: The input feature vector for the current time slice.

    Outputs:
        h_t: The updated internal state vector for the current time slice.
        y_t: The output prediction vector.

    """
    def __init__(self, config, in_dim, hidden_dim, out_dim, enable_bias=True,
                 recurrent_drop_prob=0, input_drop_prob = 0, 
                 enable_input_layernorm=True,
                 enable_state_layernorm=True,
                 no_params_layer_norm = False):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.enable_input_layernorm = enable_input_layernorm
        if no_params_layer_norm:
            self.ln = LayerNormNoParams(in_dim)
        else: 
            self.ln = nn.LayerNorm((in_dim,))
        
        self.h_prev_to_h = nn.Linear(hidden_dim, hidden_dim, bias=enable_bias)
        self.x_to_hidden_dim = nn.Linear(in_dim, hidden_dim, bias=False)
        self.h_to_y = nn.Linear(hidden_dim, out_dim, bias=enable_bias)
        self.normalize_h_prev = enable_state_layernorm
        if self.normalize_h_prev:
            # Add layernorm before hidden state update
            if no_params_layer_norm:
                self.ln_h = LayerNormNoParams(hidden_dim)
            else:
                self.ln_h = nn.LayerNorm((hidden_dim,))

        if input_drop_prob > 0:
            self.input_dropout = nn.Dropout(input_drop_prob)
        else:
            self.input_dropout = None

        if recurrent_drop_prob > 0:
            self.recurrent_dropout = nn.Dropout(recurrent_drop_prob)
        else:
            self.recurrent_dropout = None

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        W_h_h = self.h_prev_to_h.weight.data
        W_h_y = self.h_to_y.weight.data
        W_x_h = self.x_to_hidden_dim.weight.data.t()
        
        self.config.logger.debug("W_h_h: min value: {}; max value: {}".format(W_h_h.min(), W_h_h.max()))
        self.config.logger.debug("W_x_h: min value: {}; max value: {}".format(W_x_h.min(), W_x_h.max()))
        self.config.logger.debug("W_h_y: min value: {}; max value: {}".format(W_h_y.min(), W_h_y.max()))
        if do_plots:
            plot_image_matrix(W_x_h, file_name=f"{self.config.results_dir}/{description}_W_x_h.png", 
                                title=f"{description}: W_x_h")
            plot_image_matrix(W_h_h, file_name=f"{self.config.results_dir}/{description}_W_h_h.png", 
                                title=f"{description}: W_h_h")
            plot_image_matrix(W_h_y, file_name=f"{self.config.results_dir}/{description}_W_h_y.png", 
                                title=f"{description}: W_h_y")
            

    def forward(self, x, h_prev):
        """

        Run the forward pass of the RNN on a batch of inputs for the current
        time step. This will consume the input features $x_t$ (i.e., x) and
        the previous hidden state $h_{t-1}$ (i.e., h_prev) and compute the
        updated hidden state $h_t$ (i.e., output h) and the output activations
        $y_t$ (i.e., y).

        Args:
            x (torch.tensor for float): Shape (batch_size, x_dim) tensor containing
                the input features for the current time step.
            h_prev (torch.tensor for float): Shape (batch_size, h_dim) tensor containing
                the hidden state activations from the previous time step.

        Returns:
            (y, h): where y is a shape (batch_size, y_dim) tensor containing the
                predicted output activations for the current time step and
                h is a shape (batch_size, h_dim) tensor containing new hidden
                state activations for the current time step. This h will then be
                used as the `h_prev` input for the next time step.
            
        """
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        if self.enable_input_layernorm:
            x = self.ln(x)
        x_proj_to_h_dim = self.x_to_hidden_dim(x)
        
        if self.normalize_h_prev:
            h_prev = self.ln_h(h_prev)
        h_proj_to_h_dim = self.h_prev_to_h(h_prev)
        g = x_proj_to_h_dim + h_proj_to_h_dim
        h = F.gelu(g)
        #h = F.relu(g)

        if self.recurrent_dropout is not None:
            h = self.recurrent_dropout(h)
        y = self.h_to_y(h)
        return y, h


# (Not used in experiments) todo: compare against more sophisticated factorized RNNs, such as factorized LSTM, GRU, etc..
class BasicGRU(nn.Module):
    """A basic GRU.

    This implements a basic GRU for 1 time slice.

    This is slightly more optimized than the basic GRU. It fuses some
    matrix multiplications.

    Inputs:
        h_prev: The internal state vector from the previous time slice.
        x_t: The input feature vector for the current time slice.

    Outputs:
        h_t: The updated internal state vector for the current time slice.
        y_t: The output prediction vector.

    """

    def __init__(self, in_dim, hidden_dim, out_dim, enable_bias=True, 
                 recurrent_drop_prob=0, input_drop_prob = 0,
                 enable_input_layernorm=True,
                 enable_state_layernorm=True,
                 no_params_layer_norm = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_input_layernorm = enable_input_layernorm

        if no_params_layer_norm:
            self.ln = LayerNormNoParams(in_dim)
        else:
            self.ln = nn.LayerNorm((in_dim,))
        z_dim = in_dim + hidden_dim
        self.z_to_i = nn.Linear(z_dim, hidden_dim, bias=enable_bias)
        self.z_to_r = nn.Linear(z_dim, hidden_dim, bias=enable_bias)
        self.to_h_cand = nn.Linear(z_dim, hidden_dim, bias=enable_bias)
        self.h_to_y = nn.Linear(hidden_dim, out_dim, bias=enable_bias)
        self.normalize_h_prev = enable_state_layernorm
        if self.normalize_h_prev:
            # Add layernorm before hidden state update
            if no_params_layer_norm:
                self.ln_h = LayerNormNoParams(hidden_dim)
            else:
                self.ln_h = nn.LayerNorm((hidden_dim,))
        if input_drop_prob > 0:
            self.input_dropout = nn.Dropout(input_drop_prob)
        else:
            self.input_dropout = None

        if recurrent_drop_prob > 0:
            self.recurrent_dropout = nn.Dropout(recurrent_drop_prob)
        else:
            self.recurrent_dropout = None

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        pass

    def forward(self, x, h_prev):
        """

        Returns:
            (y, h): where y is the predicted output and h is the internal state
                for the current time slice.

        """
        if self.input_dropout is not None:
            x = self.input_dropout(x)
        if self.enable_input_layernorm:
            x = self.ln(x)

        (batch_size, x_dim) = x.size()
        (_, h_dim) = h_prev.size()
        if self.normalize_h_prev:
            h_prev = self.ln_h(h_prev)
        # Shape = (batch_size, z_dim)
        z = torch.cat((h_prev, x), dim=1)
        i = self.z_to_i(z)
        i = torch.sigmoid(i)
        r = self.z_to_r(z)
        r = torch.sigmoid(r)
        gated_h_prev = r * h_prev
        q = torch.cat((gated_h_prev, x), dim=1)
        h_cand = self.to_h_cand(q)
        h_cand = torch.tanh(h_cand)
        h = (1 - i) * h_prev + i * h_cand
        if self.recurrent_dropout is not None:
            h = self.recurrent_dropout(h) # reccurent dropout
        y = self.h_to_y(h)
        return y, h


# (Not used in experiments) todo: compare against more sophisticated factorized RNNs, such as factorized LSTM, GRU, etc..
class BasicLSTM(nn.Module):
    """A basic LSTM.

    This implements the LSTM for 1 time slice.

    Inputs:
        h_prev: The internal state vector from the previous time slice.
        x_t: The input feature vector for the current time slice.

    Outputs:
        h_t: The updated internal state vector for the current time slice.
        y_t: The output prediction vector.

    """

    def __init__(self, in_dim, hidden_dim, out_dim, enable_bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        no_params_layer_norm = False

        if no_params_layer_norm:
            self.ln = LayerNormNoParams(in_dim)
        else:
            self.ln = nn.LayerNorm((in_dim,))
        self.h_to_f_part = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_to_i_part = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_to_o_part = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_to_c_part = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.x_to_f_part = nn.Linear(in_dim, hidden_dim, bias=enable_bias)
        self.x_to_i_part = nn.Linear(in_dim, hidden_dim, bias=enable_bias)
        self.x_to_o_part = nn.Linear(in_dim, hidden_dim, bias=enable_bias)
        self.x_to_c_part = nn.Linear(in_dim, hidden_dim, bias=enable_bias)

        self.h_to_y_part = nn.Linear(hidden_dim, out_dim, bias=enable_bias)
        self.has_cell = True

        self.normalize_h_prev = True
        if self.normalize_h_prev:
            # Add layernorm before hidden state update
            if no_params_layer_norm:
                self.ln_h = LayerNormNoParams(hidden_dim)
            else:
                self.ln_h = nn.LayerNorm((hidden_dim,))

    def forward(self, x, c_prev, h_prev):
        """

        Args:
            x (tensor): Input tensor.

        Returns:
            (y, h): where y is the predicted output and h is the internal state
                for the current time slice.

        """
        did_unsqueeze = False
        if x.ndim == 1:
            x = torch.unsqueeze(x, 0)
            c_prev = torch.unsqueeze(c_prev, 0)
            h_prev = torch.unsqueeze(h_prev, 0)
            did_unsqueeze = True
        if self.normalize_h_prev:
            h_prev = self.ln_h(h_prev)
        h_to_f_part = self.h_to_f_part(h_prev)
        h_to_i_part = self.h_to_i_part(h_prev)
        h_to_o_part = self.h_to_o_part(h_prev)
        h_to_c_part = self.h_to_c_part(h_prev)

        x = self.ln(x)
        x_to_f_part = self.x_to_f_part(x)
        x_to_i_part = self.x_to_i_part(x)
        x_to_o_part = self.x_to_o_part(x)
        x_to_c_part = self.x_to_c_part(x)

        f_t = F.sigmoid(h_to_f_part + x_to_f_part)
        i_t = F.sigmoid(h_to_i_part + x_to_i_part)
        c_prime_t = F.tanh(h_to_c_part + x_to_c_part)
        c_t = f_t * c_prev + i_t * c_prime_t
        o_t = F.sigmoid(h_to_o_part + x_to_o_part)
        h_t = o_t * F.tanh(c_t)

        h_to_y_part = self.h_to_y_part(h_t)
        y_t = F.gelu(h_to_y_part)
        if did_unsqueeze:
            y_t  = y_t.squeeze(0)
            c_t = c_t.squeeze(0)
            h_t = h_t.squeeze(0)
        return y_t, c_t, h_t



class PFCBlock(nn.Module):
    """Predictive Factorized Coupling (PFC) block

    Model:

    y           W_y
       approx=       * h
    x           W_x

    Inference (forward) updates rules:

    Given a supplied input x, perform NMF right update rules to infer h in:

    x approx= W_x * h

    and then compute the output y as

    y = W_y * h

    This is a differentiable module and so the block can be used in arbitrary computation
    graphs and trained with backpropagation.

    Args:
        config: A struct-like structure with all needed parameters.

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        Wx = torch.rand(config.x_dim, config.basis_vector_count)*config.weights_noise_scale
        Wy = torch.rand(config.y_dim, config.basis_vector_count)*config.weights_noise_scale
        self.Wx = nn.Parameter(Wx)
        self.Wy = nn.Parameter(Wy)
        self.h_tran_copy = None
        self.lambda_1 = 0.0
        self.lambda_2 = 0.0
        self.min_val = 1e-5
        self.instance_name = "PFC"
            

    @torch.no_grad()    
    def print_stats(self, do_plots=True):
        """

        """
        logger = self.config.logger
        logger.debug("Wx: min value: {}; max value: {}".format(self.Wx.min(), self.Wx.max()))
        logger.debug("Wy: min value: {}; max value: {}".format(self.Wy.min(), self.Wy.max()))
        if self.h_tran_copy is not None:
            h_tran = self.h_tran_copy
            logger.debug("H: min value: {}; max value: {}".format(h_tran.min(), h_tran.max()))
            h_sparsity = hoyer_sparsity(h_tran, dim=1)
            mean_sparsity = torch.mean(h_sparsity).item()
            logger.debug(f"H: sparsity: {mean_sparsity}")
        
        if do_plots:
            plot_image_matrix(self.Wx, file_name=f"{self.config.results_dir}/{self.instance_name}_Wx.png", 
                                title=f"{self.instance_name}_Wx")
            plot_image_matrix(self.Wy, file_name=f"{self.config.results_dir}/{self.instance_name}_Wy.png", 
                                title=f"{self.instance_name}_Wy")
            if self.h_tran_copy is None:
                return
            plot_image_matrix(h_tran.t(), file_name=f"{self.config.results_dir}/{self.instance_name}_H.png", 
                                title=f"{self.instance_name}_H")
            images = self.Wx.transpose(0, 1)
            num_images, image_dim = images.size()
            num_to_plot = min(num_images, 100)
            if image_dim == 28*28:
                plot_rows_as_images(images, 
                                        file_name=f"{self.config.results_dir}/{self.instance_name}_Wx_as_images.png", 
                                        img_height=28, img_width=28, plot_image_count=num_to_plot)
                # Plot reconstruction, i.e., generated/predicted input x.
                reconstruction_input = torch.einsum("ij,kj->ki", self.Wx, h_tran)
                plot_rows_as_images(reconstruction_input, 
                                    file_name=f"{self.config.results_dir}/{self.instance_name}_reconstructed_input_images.png", 
                                    img_height=28, img_width=28, plot_image_count=None)
            elif image_dim == 32*32*3:
                plot_rows_as_images(
                    images,
                    file_name=f"{self.config.results_dir}/{self.instance_name}_Wx_as_images.png", 
                    img_height=32,
                    img_width=32,
                    img_channels=3,
                    plot_image_count=num_to_plot,
                    normalize=True
                )
                reconstruction_input = torch.einsum("ij,kj->ki", self.Wx, h_tran)
                plot_rows_as_images(
                    reconstruction_input,
                    file_name=f"{self.config.results_dir}/{self.instance_name}_reconstructed_input_images.png", 
                    img_height=32,
                    img_width=32,
                    img_channels=3,
                    plot_image_count=num_to_plot,
                    normalize=True
                )


    def forward(self, x, y_targets=None):
        """Forward update from port x to port y.

        Given x, iterate to a solution h and then compute y as a linear function of h.

        Args:
            x (Tensor): float tensor of shape (x_dim, batch_size) containing the input features.
            y_targets (Tensor) or None: float tensor of shape (y_dim, batch_size) containing the target features.

        Returns:
            (y_pred, loss) where
                y_pred (Tensor): float tensor of shape (y_dim, batch_size) containing the
                    class label predictions. (the predicted label will be the maximum activated
                    label index).
                loss (Tensor): float tensor of shape (1,) containing the loss. The type of
                    loss to use is specified in the `config` instance that was passed to the
                    constructor.
        
        """
        (x_dim, batch_size) = x.size()
        assert self.config.x_dim == x_dim
        h_tran_shape = (batch_size, self.config.basis_vector_count)
        h_tran = torch.rand(h_tran_shape, device=x.device) * self.config.h_noise_scale    
        Wx = self.Wx
        nmf_inference_iters_no_grad = self.config.nmf_inference_iters_no_grad

        with torch.no_grad():
            h_tran = fista_right_update(x, Wx, tolerance=self.config.fista_tolerance, 
                                            max_iterations=nmf_inference_iters_no_grad, 
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)
        h_tran = fista_right_update(x, Wx, h_tran, tolerance=self.config.fista_tolerance, 
                                            max_iterations=self.config.nmf_gradient_iters_with_grad, 
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)
        self.h_tran = h_tran
        with torch.no_grad():
            # Save a copy for viewing stats
            self.h_tran_copy = h_tran.clone().detach()
          
        y_pred = torch.einsum(
            "ij,kj->ik", self.Wy, h_tran)
        
        if y_targets is None:
            return y_pred
        else:
            if self.config.loss_type == "labels_mse":
                # Compute loss only on the label predictions
                loss = torch.nn.functional.mse_loss(y_pred, y_targets, reduction='mean')
            elif self.config.loss_type == "images_labels_mse":
                # Compute loss on label predictions and input image reconstructions
                x_pred = torch.einsum("ij,kj->ik", self.Wx, h_tran)
                loss_label_pred = torch.nn.functional.mse_loss(y_pred, y_targets, reduction='mean')
                loss_input_image_pred = torch.nn.functional.mse_loss(x_pred, x, reduction='mean')
                class_label_loss_strength = self.config.class_label_loss_strength
                loss = class_label_loss_strength*loss_label_pred + (1 - class_label_loss_strength)*loss_input_image_pred
            else:
                raise ValueError("bad value")

            return y_pred, loss

    def clip_weights_nonnegative(self):
        min_val = self.min_val
        with torch.no_grad():
            self.Wx.data = torch.clamp(self.Wx.data, min=min_val)
            self.Wy.data = torch.clamp(self.Wy.data, min=min_val)

    def normalize_weights(self):
        with torch.no_grad():
            print("normalize_weights(): not implemented")


class PFC2Layer(nn.Module):
    """Fully-connected residual network with 2 PFC blocks

    This implements two PFC blocks connected in serial.

    Model: 2 of the following in serial:

    y           W_y
       approx=       * h
    x           W_x

    where the first PFC block has a residual connection.
    

    Args:
        config: A struct-like structure with all needed parameters.

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.min_val = 1e-5
        # Block 1 parameters
        Wx = torch.rand(config.x_dim, config.basis_vector_count1)*config.weights_noise_scale
        Wy = torch.rand(config.hidden_dim, config.basis_vector_count1)*config.weights_noise_scale
        self.Wx = nn.Parameter(Wx)
        self.Wy = nn.Parameter(Wy)
        # Block 2 parameters
        Wx2 = torch.rand(config.hidden_dim, config.basis_vector_count2)*config.weights_noise_scale
        Wy2 = torch.rand(config.y_dim, config.basis_vector_count2)*config.weights_noise_scale
        self.Wx2 = nn.Parameter(Wx2)
        self.Wy2 = nn.Parameter(Wy2)
        self.lambda_1 = 0.0
        self.lambda_2 = 0.0
        self.instance_name = "PFC"
        if config.nmf_inference_algorithm == "sgd":
            self.inference_sgd_lr = nn.Parameter(torch.tensor(config.initial_inference_sgd_lr))
            

    @torch.no_grad()    
    def print_stats(self, do_plots=True):
        """Print debug info.

        """
        if self.config.nmf_inference_algorithm == "sgd":
            print(f"sgd learning rate: {self.inference_sgd_lr.item()}")

        # Layer 1:
        print("Wx: min value: {}; max value: {}".format(self.Wx.min(), self.Wx.max()))
        print("Wy: min value: {}; max value: {}".format(self.Wy.min(), self.Wy.max()))
        h_tran = self.h_tran_copy
        print("H: min value: {}; max value: {}".format(h_tran.min(), h_tran.max()))
        h_sparsity = hoyer_sparsity(h_tran, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        print("H: sparsity: ", mean_sparsity)

        # Layer 2:
        print("Wx2: min value: {}; max value: {}".format(self.Wx2.min(), self.Wx2.max()))
        print("Wy2: min value: {}; max value: {}".format(self.Wy2.min(), self.Wy2.max()))
        h2_tran = self.h2_tran_copy
        print("H2: min value: {}; max value: {}".format(h2_tran.min(), h2_tran.max()))
        h2_sparsity = hoyer_sparsity(h2_tran, dim=1)
        mean_sparsity2 = torch.mean(h2_sparsity).item()
        print("H2: sparsity: ", mean_sparsity2)
        
        if do_plots:
            # Layer 1:
            plot_image_matrix(self.Wx, file_name=f"{self.config.results_dir}/{self.instance_name}_Wx.png", 
                                title=f"{self.instance_name}_Wx")
            plot_image_matrix(self.Wy, file_name=f"{self.config.results_dir}/{self.instance_name}_Wy.png", 
                                title=f"{self.instance_name}_Wy")
            plot_image_matrix(h_tran.t(), file_name=f"{self.config.results_dir}/{self.instance_name}_H.png", 
                                title=f"{self.instance_name}_H")
            
            # Layer 2:
            plot_image_matrix(self.Wx2, file_name=f"{self.config.results_dir}/{self.instance_name}_Wx2.png", 
                                title=f"{self.instance_name}_Wx2")
            plot_image_matrix(self.Wy2, file_name=f"{self.config.results_dir}/{self.instance_name}_Wy2.png", 
                                title=f"{self.instance_name}_Wy2")
            plot_image_matrix(h2_tran.t(), file_name=f"{self.config.results_dir}/{self.instance_name}_H2.png", 
                                title=f"{self.instance_name}_H2")
            images_Wx = self.Wx.transpose(0, 1)
            images_Wx2 = self.Wx2.transpose(0, 1)
            num_images, image_dim = images_Wx.size()
            num_to_plot = min(num_images, 100)
            if image_dim == 28*28:
                plot_rows_as_images(images_Wx, 
                                        file_name=f"{self.config.results_dir}/{self.instance_name}_Wx_as_images.png", 
                                        img_height=28, img_width=28, plot_image_count=num_to_plot, 
                                        normalize=False)
                plot_rows_as_images(images_Wx2, 
                                        file_name=f"{self.config.results_dir}/{self.instance_name}_Wx2_as_images.png", 
                                        img_height=28, img_width=28, plot_image_count=num_to_plot, 
                                        normalize=False)
                # Plot reconstruction, i.e., generated/predicted input x.
                reconstruction_input = torch.einsum("ij,kj->ki", self.Wx, h_tran)
                plot_rows_as_images(reconstruction_input, 
                                    file_name=f"{self.config.results_dir}/{self.instance_name}_reconstructed_input_images.png", 
                                    img_height=28, img_width=28, plot_image_count=None,
                                    normalize=False)
            elif image_dim == 32*32*3:
                plot_rows_as_images(
                    images_Wx,
                    file_name=f"{self.config.results_dir}/{self.instance_name}_Wx_as_images.png", 
                    img_height=32,
                    img_width=32,
                    img_channels=3,
                    plot_image_count=num_to_plot,
                    normalize=True
                )
                plot_rows_as_images(
                    images_Wx2,
                    file_name=f"{self.config.results_dir}/{self.instance_name}_Wx2_as_images.png", 
                    img_height=32,
                    img_width=32,
                    img_channels=3,
                    plot_image_count=num_to_plot,
                    normalize=True
                )
                reconstruction_input = torch.einsum("ij,kj->ki", self.Wx, h_tran)
                plot_rows_as_images(
                    reconstruction_input,
                    file_name=f"{self.config.results_dir}/{self.instance_name}_reconstructed_input_images.png", 
                    img_height=32,
                    img_width=32,
                    img_channels=3,
                    plot_image_count=num_to_plot,
                    normalize=True
                )


    def forward(self, x, y_targets):
        """Forward update from port x to port y.

        Given x, iterate to a solution h and then compute y as a linear function of h.

        Args:
            x (Tensor): float tensor of shape (x_dim, batch_size) containing the input features.
            y_targets (Tensor): Int tensor of shape (y_dim, batch_size) containing the target features.

        Returns:
            (y_pred, loss) where
                y_pred (Tensor): float tensor of shape (y_dim, batch_size) containing the
                    class label predictions. (the predicted label will be the maximum activated
                    label index).
                loss (Tensor): float tensor of shape (1,) containing the loss. The type of
                    loss to use is specified in the `config` instance that was passed to the
                    constructor.
        
        """
        (x_dim, batch_size) = x.size()
        assert self.config.x_dim == x_dim
        # Block 1:
        h_tran_shape = (batch_size, self.config.basis_vector_count1)
        h_tran = torch.rand(h_tran_shape, device=x.device) * self.config.h_noise_scale    
        Wx = self.Wx
        
        if self.training and self.config.nmf_is_randomized_iters_no_grad:
            random_integer = random.randint(0, self.config.nmf_inference_iters_no_grad)
            nmf_inference_iters_no_grad = random_integer
        else:
            nmf_inference_iters_no_grad = self.config.nmf_inference_iters_no_grad

        with torch.no_grad():
            tolerance = self.config.fista_tolerance
            h_tran = fista_right_update(x, Wx, tolerance=tolerance, 
                                            max_iterations=nmf_inference_iters_no_grad,
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)
        h_tran = fista_right_update(x, Wx, h_tran, tolerance=tolerance, 
                                            max_iterations=self.config.nmf_gradient_iters_with_grad,
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)        
        self.h_tran = h_tran
        with torch.no_grad():
            # Save a copy for viewing stats
            self.h_tran_copy = h_tran.clone().detach()
          
        y_pred = torch.einsum(
            "ij,kj->ik", self.Wy, h_tran)

        # Block 2:
        h2_tran_shape = (batch_size, self.config.basis_vector_count2)
        h2_tran = torch.rand(h2_tran_shape, device=x.device) * self.config.h_noise_scale    
        Wx2 = self.Wx2

        # Enable skip connection.
        #y_pred = x - y_pred # skip connection. This is fine for semi-NMF.
        #y_pred = x * y_pred # also works
        y_pred = torch.clamp(x - y_pred, min=self.min_val) # note: relu() prevents negative values
        
        with torch.no_grad():
            tolerance = self.config.fista_tolerance
            h2_tran = fista_right_update(y_pred, Wx2, tolerance=tolerance, 
                                            max_iterations=nmf_inference_iters_no_grad,
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)
        h2_tran = fista_right_update(y_pred, Wx2, h2_tran, tolerance=tolerance, 
                                            max_iterations=self.config.nmf_gradient_iters_with_grad,
                                            apply_normalization_scaling=self.config.enable_h_column_normalization,
                                            debug=False)
        self.h2_tran = h2_tran
        with torch.no_grad():
            # Save a copy for viewing stats
            self.h2_tran_copy = h2_tran.clone().detach()
          
        y2_pred = torch.einsum(
            "ij,kj->ik", self.Wy2, h2_tran)

        if self.config.loss_type == "labels_mse":
            # Compute loss only on the label predictions
            loss = torch.nn.functional.mse_loss(y2_pred, y_targets, reduction='mean')
        elif self.config.loss_type == "images_labels_mse":
            # Comptue loss on label precitions and input image reconstructions
            x_pred = torch.einsum("ij,kj->ik", self.Wx, h_tran)
            loss_label_pred = torch.nn.functional.mse_loss(y2_pred, y_targets, reduction='mean')
            loss_input_image_pred = torch.nn.functional.mse_loss(x_pred, x, reduction='mean')

            y_pred_reconstruction = torch.einsum("ij,kj->ik", Wx2, h2_tran)
            loss_hidden_layer_pred = torch.nn.functional.mse_loss(y_pred_reconstruction, y_pred, reduction='mean')

            loss = loss_label_pred + loss_input_image_pred + loss_hidden_layer_pred
        else:
            raise ValueError("bad value")
        return y2_pred, loss

    def clip_weights_nonnegative(self):
        min_val = self.min_val
        with torch.no_grad():
            # Block 1:
            self.Wx.data = torch.clamp(self.Wx.data, min=min_val)
            self.Wy.data = torch.clamp(self.Wy.data, min=min_val)
            # Block 2:
            self.Wx2.data = torch.clamp(self.Wx2.data, min=min_val)
            self.Wy2.data = torch.clamp(self.Wy2.data, min=min_val)


class FactorizedRNN(nn.Module):
    """A factorized RNN.

    This uses algorithm unrolling and is intended to be trained 
    with the usual backpropagation.

    Note: This can be used in place of conventional RNNs such as
    `VanillaRNN` since the `forward()` method is compatible.

    For forward() method computes the updated hidden state h and output y
    for one time slice only.

    The model for all slices corresponds to the following factorization:

    Input sequence of feature vectors: [x_0  x_1 x_2 ... x_T-1]
    Hidden state vectors: [h_-1 h_0 h_1 ... h_T-1]
    Target output sequence of vectors: [y_0  y_1 y_2 ... y_T-1]

    General factorization:

    [Y]      = [y_0  y_1 y_2 ... y_T-1]
    [H_prev] = [h_-1 h_0 h_1 ... h_T-2] approx= W [h_0 h_1 h_2 ... h_T-1]
    [X]      = [x_0  x_1 x_2 ... x_T-1]

        [W_y]
    W = [W_h]
        [W_x]

    W has size (x_dim + hidden_state_dim + y_dim, basis_vector_count).

    where hidden_state_dim = basis_vector_count, making W_h a square matrix.
    
    """
    def __init__(self, config, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        if config.enforce_nonneg_params:
            W_y = torch.rand(out_dim, hidden_dim)*config.weights_noise_scale
            W_h = torch.rand(hidden_dim, hidden_dim)*config.weights_noise_scale
            W_x = torch.rand(in_dim, hidden_dim)*config.weights_noise_scale
        else:
            W_y = (torch.rand(out_dim, hidden_dim) - 0.5)*config.weights_noise_scale
            W_h = (torch.rand(hidden_dim, hidden_dim) - 0.5)*config.weights_noise_scale
            W_x = (torch.rand(in_dim, hidden_dim) - 0.5)*config.weights_noise_scale
        self.W_y = nn.Parameter(W_y)
        self.W_h = nn.Parameter(W_h)
        self.W_x = nn.Parameter(W_x)

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        self.config.logger.debug("W_y: min value: {}; max value: {}".format(self.W_y.min(), self.W_y.max()))
        self.config.logger.debug("W_h: min value: {}; max value: {}".format(self.W_h.min(), self.W_h.max()))
        self.config.logger.debug("W_x: min value: {}; max value: {}".format(self.W_x.min(), self.W_x.max()))
        if do_plots:
            plot_image_matrix(self.W_y, file_name=f"{self.config.results_dir}/{description}_W_y.png", 
                                title=f"{description}: W_y")
            plot_image_matrix(self.W_h, file_name=f"{self.config.results_dir}/{description}_W_h.png", 
                                title=f"{description}: W_h")
            plot_image_matrix(self.W_x, file_name=f"{self.config.results_dir}/{description}_W_x.png", 
                                title=f"{description}: W_x")

    def clip_weights_nonnegative(self):
        with torch.no_grad():
            min_val = 1e-5
            self.W_y[self.W_y < min_val] = min_val
            self.W_h[self.W_h < min_val] = min_val
            self.W_x[self.W_x < min_val] = min_val
            

    def forward(self, x, h_prev, return_reconstruction_loss = False):
        """

        Run the forward pass of the RNN on a batch of inputs for the current
        time step. This will consume the input features $x_t$ (i.e., x) and
        the previous hidden state $h_{t-1}$ (i.e., h_prev) and compute the
        updated hidden state $h_t$ (i.e., output h) and the output activations
        $y_t$ (i.e., y).

        Args:
            x (torch.tensor for float): Shape (batch_size, x_dim) tensor containing
                the input features for the current time step.
            h_prev (torch.tensor for float): Shape (batch_size, h_dim) tensor containing
                the hidden state activations from the previous time step.
            return_reconstruction_loss (boolean): If True, return the reconstruction loss for the input
                h_prev and x.

        Returns:
            If `return_reconstruction_loss` is False:
            (y, h): where y is a shape (batch_size, y_dim) tensor containing the
                predicted output activations for the current time step and
                h is a shape (batch_size, h_dim) tensor containing new hidden
                state activations for the current time step. This h will then be
                used as the `h_prev` input for the next time step.
            If `return_reconstruction_loss` is True:
            (y, h, loss_x, loss_h_prev): where y is a shape (batch_size, y_dim) tensor containing the
                predicted output activations for the current time step and
                h is a shape (batch_size, h_dim) tensor containing new hidden
                state activations for the current time step. This h will then be
                used as the `h_prev` input for the next time step.
                `loss_x` is the reconstruction loss for the observed input `x`.
                `loss_h_prev` is the reconstruction loss for the input previous state `h_prev`.

            
        """
        (batch_size, x_dim) = x.size()
        (_, h_dim) = h_prev.size()
        # Shape = (batch_size, z_dim)
        z = torch.cat((h_prev, x), dim=1)
        # weights
        W_z = torch.cat((self.W_h, self.W_x), dim=0)
        assert W_z.size() == (self.hidden_dim + x_dim, self.hidden_dim)

        # Factorization: Z approx= W_z * H 
        Z_temp = z.t()
        h = h_prev
                
        with torch.no_grad():
            h = fista_right_update(Z_temp, W_z, h, tolerance=self.config.fista_tolerance, 
                                                max_iterations=self.config.nmf_inference_iters_no_grad,
                                                shrinkage_sparsity=self.config.sparsity_L1_H,
                                                apply_normalization_scaling=True,
                                                logger=self.config.logger)

        h = fista_right_update(Z_temp, W_z, h, tolerance=self.config.fista_tolerance, 
                                            max_iterations=self.config.nmf_gradient_iters_with_grad,
                                            shrinkage_sparsity=self.config.sparsity_L1_H,
                                            apply_normalization_scaling=True,
                                            logger=self.config.logger)
        
        # Predictions based on inferred H:
        # shape: (batch_size, y_dim)
        y = torch.einsum("ij,kj->ki", self.W_y, h)

        if return_reconstruction_loss:
            # Also return the input reconstruction loss for x and h_prev.
            h_prev_pred = torch.einsum("ij,kj->ki", self.W_h, h)
            h_prev_targets = h_prev.clone().detach()
            loss_h_prev = torch.nn.functional.mse_loss(h_prev_pred, h_prev_targets, reduction='sum')/torch.numel(h_prev_targets)

            x_pred = torch.einsum("ij,kj->ki", self.W_x, h)
            x_targets = x.clone().detach()
            loss_x = torch.nn.functional.mse_loss(x_pred, x_targets, reduction='sum')/torch.numel(x_targets)
            return y, h, loss_x, loss_h_prev
        else:
            return y, h


class FactorizedRNNWithoutBackprop:
    """Factorized RNN using NMF learning updates (no backpropagation).

    This is the factorized RNN as described in the paper.
    This version uses conventional NMF updates (no unrolling), similar to
    the RNNs in the "Positive Factor Networks" paper. We
    iterate right (inference) updates until convergence and then perform a left 
    (learning) update. As a result, the model is trained without backpropagation.

    Input sequence of feature vectors: [x_0  x_1 x_2 ... x_T-1]
    Hidden state vectors: [h_-1 h_0 h_1 ... h_T-1]
    Target output sequence of vectors: [y_0  y_1 y_2 ... y_T-1]

    General factorization:

    [Y]      = [y_0  y_1 y_2 ... y_T-1]
    [H_prev] = [h_-1 h_0 h_1 ... h_T-2] approx= W [h_0 h_1 h_2 ... h_T-1]
    [X]      = [x_0  x_1 x_2 ... x_T-1]

        [W_y]
    W = [W_h]
        [W_x]

    W has size (x_dim + hidden_state_dim + y_dim, basis_vector_count).

    where hidden_state_dim = basis_vector_count, making W_h a square matrix.

    Notes:
    For the case of an autoregressive model (i.e., predict the next feature
    vector x_t), Y is takes on the value of the next time slice of the input:

    [Y] = [x_1 x_2 x_3 ... x_T]
        
    For more details, see paper.

    """

    def __init__(self, config):
        self.config = config
        self.device = config.device
        if config.enforce_nonneg_params:
            self.W_y = torch.rand(config.y_dim, config.basis_vector_count)*config.weights_noise_scale
            self.W_h = torch.rand(config.basis_vector_count, config.basis_vector_count)*config.weights_noise_scale
            self.W_x = torch.rand(config.x_dim, config.basis_vector_count)*config.weights_noise_scale
        else:
            self.W_y = (torch.rand(config.y_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_h = (torch.rand(config.basis_vector_count, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_x = (torch.rand(config.x_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
        self.W_y = self.W_y.to(self.device)
        self.W_h = self.W_h.to(self.device)
        self.W_x = self.W_x.to(self.device)
        self.h_tran = None
        self.inference_noise_scale = 0
        self.min_val=1e-5

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        """

        """
        self.config.logger.debug(f"Model info for: {description}")
        
        self.config.logger.debug(f"sgd learning rate: {self.inference_sgd_lr}")
        self.config.logger.debug("W_y: min value: {}; max value: {}".format(self.W_y.min(), self.W_y.max()))
        self.config.logger.debug("W_h: min value: {}; max value: {}".format(self.W_h.min(), self.W_h.max()))
        self.config.logger.debug("W_x: min value: {}; max value: {}".format(self.W_x.min(), self.W_x.max()))
        h_tran = self.h_tran
        if h_tran is None:
            return
        self.config.logger.debug("H: min value: {}; max value: {}".format(h_tran.min(), h_tran.max()))
        h_sparsity = hoyer_sparsity(h_tran, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        self.config.logger.debug(f"H: sparsity: {mean_sparsity}")
        H = h_tran.t()
        if do_plots:
            plot_image_matrix(self.W_y, file_name=f"{self.config.results_dir}/{description}_W_y.png", 
                                title=f"{description}: W_y")
            plot_image_matrix(self.W_h, file_name=f"{self.config.results_dir}/{description}_W_h.png", 
                                title=f"{description}: W_h")
            plot_image_matrix(self.W_x, file_name=f"{self.config.results_dir}/{description}_W_x.png", 
                                title=f"{description}: W_x")
            plot_image_matrix(H, file_name=f"{self.config.results_dir}/{description}_H.png", 
                                title=f"{description}: H")
            W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
            V_pred = W @ H
            plot_image_matrix(V_pred, file_name=f"{self.config.results_dir}/{description}_V_pred.png", 
                                title=f"{description}: V_pred")
            y_pred = self.W_y @ H
            plot_image_matrix(
            y_pred, f"{self.config.results_dir}/{description}_y_pred.png", 
                title=f"{description}: Predicted values for training targets"
    )

    @torch.no_grad()
    def update_weights(self):
        """Perform the learning update using the NMF left-update step.

        We use SGD to implement the left-update step.
        """
        W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
        # Prepare X:
        H_tran = self.h_tran
        assert H_tran is not None, "You must call forward() before update_weights()"
        w_lr = fista_compute_lr_from_weights(H_tran)*self.config.w_lr_scale
        rows_W_y = self.config.y_dim
        rows_W_h = self.config.basis_vector_count
        row_start_W_h = rows_W_y + rows_W_h
        rows_W_x, t_slice_count = self.X_targets.size()
        
        V_targets = torch.zeros((rows_W_y + rows_W_h + rows_W_x, t_slice_count), device=self.device)
        V_targets[:rows_W_y,:] = self.y_targets[:, :]
        V_targets[rows_W_y:row_start_W_h, :] = self.h_prev_targets[:, :]
        V_targets[row_start_W_h:, :] = self.X_targets[:, :]

        
        W = normalize_columns_at_most_equal_max_val(W, H_tran)

        W = left_update_nmf_sgd(V_targets, 
                                W,
                                H_tran,
                                learn_rate=w_lr,
                                shrinkage_sparsity=0,
                                weight_decay=self.config.weight_decay_W,
                                min_val=self.min_val,
                                force_nonnegative=True,
                                )

        self.W_y = W[:rows_W_y, :]
        self.W_h = W[rows_W_y:row_start_W_h, :]
        self.W_x = W[row_start_W_h:, :]

    
    @torch.no_grad()
    def forward(self, X, targets = None, h_prev_init = None):
        """Forward update.

        Compute the forward update on the RNN factorizations and return
        the loss.

        Args:
            x (tensor): float tensor of shape (x_dim, seq_len)
            targets: (None or Tensor):  float tensor of shape (y_dim, seq_len). This is the
                target sequence for y_t.
            h_prev: (tensor): tensor of (size hidden_dim, 1)

        Returns:
            y (tensor): float tensor of shape (y_dim, seq_len)
            h_next (tensor): Size is  (size hidden_dim, 1)
            loss (tensor): tensor of size 1. This is only returned if `targets` is not None.


        """
        (x_dim, seq_len) = X.size()
        if h_prev_init is not None:
            H_prev = torch.rand(self.config.basis_vector_count, seq_len-1, device=X.device)*self.inference_noise_scale
            H_prev = torch.cat((h_prev_init, H_prev), dim=1)
        else:
            #H_prev = torch.zeros(self.config.basis_vector_count, seq_len, device=X.device)
            H_prev = torch.rand(self.config.basis_vector_count, seq_len, device=X.device)*self.inference_noise_scale
        
        h_tran = torch.rand(seq_len, self.config.basis_vector_count, device=X.device)*self.inference_noise_scale

        # inference weights
        W_z = torch.cat((self.W_h, self.W_x), dim=0)
        # Factorization: Z approx= W_z * H
        
        # Iterate to convergence on each time slice, then use the converged state output
        # as the input for the next time slice.
        lr = fista_compute_lr_from_weights(W_z)*self.config.h_lr_scale
        self.inference_sgd_lr = lr
        y_pred_list = list()
        loss = 0
        # let's set the maximum allowable value for hidden states to be equal to the largest value
        # in X or W_z:
        max_allowed_value = X.max()
            
        for t in range(seq_len):
            h_prev_col_vec =  H_prev[:, t].unsqueeze(1)
            x_col_vec = X[:, t].unsqueeze(1)
            z_col_vec = torch.cat((h_prev_col_vec, x_col_vec), dim=0)
            h_tran_row_vec = h_tran[t, :].unsqueeze(0)
              
            for i in range(self.config.nmf_inference_iters):
                h_tran_row_vec = right_update_nmf_sgd(
                            z_col_vec,
                            W_z,
                            h_tran_row_vec,
                            learn_rate=lr,
                            shrinkage_sparsity=self.config.sparsity_L1_H,
                            weight_decay=self.config.weight_decay_H,
                            force_nonnegative_H=True)
                                
                max_allowed_value = z_col_vec.max()
                maxh = h_tran_row_vec.max()
                if maxh > max_allowed_value:
                    # Normalize hidden state to its value does not exceed maximum input value in X.
                    h_tran_row_vec = h_tran_row_vec * (max_allowed_value / maxh)

            # Add the current inference result to h_tran:
            h_tran[t, :] = h_tran_row_vec.clone()[0, :]
            if t < seq_len - 1:
                h_next_col_vec = h_tran_row_vec.clone().t()
                H_prev[:, t+1] = h_next_col_vec[:, 0]

            # Predictions based on inferred H:
            y_pred_col_vec = torch.einsum(
                    "ij,kj->ik", self.W_y, h_tran_row_vec)
            y_pred_list.append(y_pred_col_vec)
            h_prev_pred_col_vec = torch.einsum(
                    "ij,kj->ik", self.W_h, h_tran_row_vec)
            x_pred_col_vec = torch.einsum(
                    "ij,kj->ik", self.W_x, h_tran_row_vec)

            if targets is not None:
                # Update loss
                # Y pred vs targets:
                target_col_vec = targets[:, t].unsqueeze(1)
                loss_y = torch.nn.functional.mse_loss(y_pred_col_vec, target_col_vec, reduction='sum')/torch.numel(target_col_vec)
                # H pred vs targets:
                loss_h = torch.nn.functional.mse_loss(h_prev_pred_col_vec, h_prev_col_vec, reduction='sum')/torch.numel(h_prev_col_vec)
                # X pred vs targets:
                loss_x = torch.nn.functional.mse_loss(x_pred_col_vec, x_col_vec, reduction='sum')/torch.numel(x_col_vec)
                loss += loss_y + loss_h + loss_x

        # Save a copy for viewing stats
        self.h_tran = h_tran
        loss = loss/seq_len
        y_pred = torch.cat(y_pred_list, dim=1)
        h_next = h_tran_row_vec.t()
        # Save targets:
        self.y_targets = targets
        self.h_prev_targets = H_prev
        self.X_targets = X
        if targets is not None:
            return y_pred, h_next, loss
        else:
            return y_pred, h_next
            
        
    @torch.no_grad()
    def generate_from_seed(self, seed_seq, generate_frame_count, make_plots=True):
        """Generate a sequence from a starting seed sequence.

        Args:
            seed_seq (tensor): Shape (x_dim, seed_len) tensor containing the seed
                sequence.
            generate_frame_count (int): Length of sequence to generate after the seed.

        Returns:
            (tensor): The generated sequence, including the seed.
        """
        (x_dim, seed_len) = seed_seq.size()
        assert seed_len > 1
        seq_len = seed_len + generate_frame_count
        # x is the concatenation of the seed_seq and x_gen.
        x_gen = torch.zeros((x_dim, generate_frame_count), dtype=seed_seq.dtype, 
                              device=seed_seq.device)
        X = torch.cat((seed_seq, x_gen), dim=1)
        # y is the concatenation of y_seed = seed_seq (excluding the first time slice) and y_gen.
        y_seed = seed_seq[:, 1:]
        y_gen = torch.zeros((x_dim, generate_frame_count + 1), dtype=seed_seq.dtype, 
                              device=seed_seq.device)
        Y = torch.cat((y_seed, y_gen), dim=1)
        # Initialize the hidden state sequence:
        # Note the the hidden state dimension must equal the number of basis vectors in W.
        H_prev = torch.rand(self.config.basis_vector_count, seq_len, device=X.device)*self.inference_noise_scale
        H_prev[:, 0] = 0
        # Now iterate to solve for the hidden states that are most consistent with the observed sequence.
        Z = torch.cat((H_prev, X), dim=0)
        # weights
        W_z = torch.cat((self.W_h, self.W_x), dim=0)
        # Factorization: Z approx= W_z * H
        h_tran = torch.rand(seq_len, self.config.basis_vector_count, device=seed_seq.device) * self.inference_noise_scale
        if make_plots:
            # Let's plot the current matrices of the factorization V approx= W * H before performing inference. 
            V = torch.cat((Y, H_prev, X), dim=0)
            #plot_image_matrix(V, file_name=f"{self.config.results_dir}/before_gen_V.png", 
            #                    title="V")
            rows_W_y = self.config.y_dim
            rows_W_h = self.config.basis_vector_count
            row_start_W_h = rows_W_y + rows_W_h
            V_init_y = V[:rows_W_y,:]
            V_init_h = V[rows_W_y:row_start_W_h, :]
            V_init_x = V[row_start_W_h:, :]
            plt.figure()
            add_colorbar = True
            ax1 = plt.subplot(3, 1, 1)
            
            im1 = plt.imshow(V_init_y, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_init_y.min(),
                vmax=V_init_y.max())
            plt.title(r'Initialized $Y$')
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im1, ax=ax1)

            ax2 = plt.subplot(3, 1, 2)
            
            im2 = plt.imshow(V_init_h, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_init_h.min(),
                vmax=V_init_h.max())
            plt.title(r'Initialized $H_{prev}$')
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im2, ax=ax2)

            ax3 = plt.subplot(3, 1, 3)
            
            im3 = plt.imshow(V_init_x, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_init_x.min(),
                vmax=V_init_x.max())
            plt.title(r'Initialized $X$')
            if add_colorbar:
                plt.colorbar(im3, ax=ax3)

            plt.tight_layout()

            # Save the figure as a PNG file
            plt.savefig(f"{self.config.results_dir}/before_gen_V.png", dpi=300)


            W =  torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
            plot_image_matrix(W, file_name=f"{self.config.results_dir}/learned_W.png", 
                                title="Learned weights: W")
            H = h_tran.t()
            plot_image_matrix(H, file_name=f"{self.config.results_dir}/before_gen_H.png", 
                                title="H")

            # Also save the weights as a subplot containing the 3 sub-matrices:
            plt.figure()
            add_colorbar = True
            ax1 = plt.subplot(3, 1, 1)
            W_y_to_plot = self.W_y
            im1 = plt.imshow(W_y_to_plot, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=W_y_to_plot.min(),
                vmax=W_y_to_plot.max())
            plt.title(r'$W_y$')
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im1, ax=ax1)

            ax2 = plt.subplot(3, 1, 2)
            W_h_to_plot = self.W_h
            im2 = plt.imshow(W_h_to_plot, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=W_h_to_plot.min(),
                vmax=W_h_to_plot.max())
            plt.title(r'$W_h$')
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im2, ax=ax2)

            ax3 = plt.subplot(3, 1, 3)
            W_x_to_plot = self.W_x
            im3 = plt.imshow(W_x_to_plot, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=W_x_to_plot.min(),
                vmax=W_x_to_plot.max())
            plt.title(r'$W_x$')
            if add_colorbar:
                plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.savefig(f"{self.config.results_dir}/learned_W_as_subplots.png", dpi=300)
        # Iterate to convergence on each time slice, then use the converged state outptut
        # as the input for the next time slice. 
        lr = fista_compute_lr_from_weights(W_z)
        max_allowed_value = seed_seq.max()
        for t in range(seq_len):           
            h_prev_col_vec =  H_prev[:, t].unsqueeze(1)
            # If x_t is unobserved, update it
            if t >= seed_len:
                # copy the previous y_t-1 into the current x_t.
                prev_y_col_vec = Y[:, t-1].unsqueeze(1)
                X[:, t] = prev_y_col_vec[:, 0]
            x_col_vec = X[:, t].unsqueeze(1)
            z_col_vec = torch.cat((h_prev_col_vec, x_col_vec), dim=0)    
                
            h_tran_row_vec = h_tran[t, :].unsqueeze(0)
            # First iterate to convergence on the current slice before doing anything else.
            for i in range(self.config.evaluation_nmf_inference_iteration_count):
                h_tran_row_vec = right_update_nmf_sgd(
                            z_col_vec,
                            W_z,
                            h_tran_row_vec,
                            learn_rate=lr,
                            shrinkage_sparsity=self.config.sparsity_L1_H,
                            force_nonnegative_H=True,
                )
                if True:
                    maxh = h_tran_row_vec.max()
                    if maxh > max_allowed_value:
                        # Normalize hidden state to its value does not exceed maximum input value in X.
                        h_tran_row_vec = h_tran_row_vec * (max_allowed_value / maxh)

            # If y_t is unobserved, update it.  
            if t >= seed_len - 1:
                y_col_vec = self.W_y @ h_tran_row_vec.t()
                Y[:, t] = y_col_vec[:, 0]

            # Add the current inference result to h_tran:
            h_tran[t, :] = h_tran_row_vec[0, :]
            if t < seq_len - 1:
                # Enforce that duplicate states in Z and H are equal.
                # Copy the converged h_t into the next time step in V.
                h_next_col_vec = h_tran_row_vec.t()
                H_prev[:, t+1] = h_next_col_vec[:, 0]
            
        # Save a copy for viewing stats
        self.h_tran = h_tran
        
        if make_plots:
            # Plot the generated sequence and corresponding factor matrices.
            # (we already plotted W, which did not change, so skip that)
            H = h_tran.t()
            plot_image_matrix(H, file_name=f"{self.config.results_dir}/after_gen_H.png", 
                                title="H")
            # Generate the reconstruction V.
            W =  torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
            V_pred = W @ H
            rows_W_y = self.config.y_dim
            rows_W_h = self.config.basis_vector_count
            row_start_W_h = rows_W_y + rows_W_h
            V_pred_y = V_pred[:rows_W_y,:]
            V_pred_h = V_pred[rows_W_y:row_start_W_h, :]
            V_pred_x = V_pred[row_start_W_h:, :]
            plt.figure()
            add_colorbar = True
            ax1 = plt.subplot(3, 1, 1)
            
            im1 = plt.imshow(V_pred_y, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_pred_y.min(),
                vmax=V_pred_y.max())
            plt.title(r'Predicted $Y$')
            ax1.set_xticks([])
            ax1.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im1, ax=ax1)

            ax2 = plt.subplot(3, 1, 2)
            
            im2 = plt.imshow(V_pred_h, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_pred_h.min(),
                vmax=V_pred_h.max())
            plt.title(r'Predicted $H_{prev}$')
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            if add_colorbar:
                plt.colorbar(im2, ax=ax2)

            ax3 = plt.subplot(3, 1, 3)
            
            im3 = plt.imshow(V_pred_x, origin="upper", cmap=plt.cm.hot, aspect="auto", 
                interpolation="nearest",
                vmin=V_pred_x.min(),
                vmax=V_pred_x.max())
            plt.title(r'Predicted $X$')
            if add_colorbar:
                plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.savefig(f"{self.config.results_dir}/generated_V.png", dpi=300)

        return X
    

class FactorizedRNNCopyTaskWithoutBackprop:
    """Factorized RNN for the Copy Task using NMF learning updates (no backpropagation).

    This version uses NMF learning updates to update the weights W. We
    iterate right (inference) updates until convergence and then perform a left 
    (learning) update. Backpropagation is not used.

    This version supports mini-batched processing. We simply extend the non-batched version to
    support batched operation by appending a batch dimension to V and H.
    
    Input sequence of feature vectors: [x_0  x_1 x_2 ... x_T-1]
    Hidden state vectors: [h_-1 h_0 h_1 ... h_T-1]
    Target output sequence of vectors: [y_0  y_1 y_2 ... y_T-1]

    RNN factorization of the form: V approx= W * H:

    [Y]      = [y_0  y_1 y_2 ... y_T-1]
    [H_prev] = [h_-1 h_0 h_1 ... h_T-2] approx= W [h_0 h_1 h_2 ... h_T-1]
    [X]      = [x_0  x_1 x_2 ... x_T-1]

        [W_y]
    W = [W_h]
        [W_x]

    W has size (x_dim + hidden_state_dim + y_dim, basis_vector_count).

    where hidden_state_dim = basis_vector_count, making W_h a square matrix.

    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.copy_seq_len = config.rand_seq_len
        if config.enforce_nonneg_params:
            self.W_y = torch.rand(config.y_dim, config.basis_vector_count)*config.weights_noise_scale
            self.W_h = torch.rand(config.basis_vector_count, config.basis_vector_count)*config.weights_noise_scale
            self.W_x = torch.rand(config.x_dim, config.basis_vector_count)*config.weights_noise_scale
        else:
            self.W_y = (torch.rand(config.y_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_h = (torch.rand(config.basis_vector_count, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_x = (torch.rand(config.x_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
        self.W_y = self.W_y.to(device)
        self.W_h = self.W_h.to(device)
        self.W_x = self.W_x.to(device)
        self.h_tran = None
        self.inference_noise_scale = 0
        self.min_val=1e-5

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        """Print debug info and plots.

        """
        logger = self.config.logger
        logger.debug(f"Model info for: {description}")
        if self.config.nmf_inference_algorithm == "sgd":
            if isinstance(self.inference_sgd_lr, torch.Tensor):
                logger.debug(f"sgd learning rate: {self.inference_sgd_lr.item()}")
        logger.debug("W_y: min value: {}; max value: {}".format(self.W_y.min(), self.W_y.max()))
        logger.debug("W_h: min value: {}; max value: {}".format(self.W_h.min(), self.W_h.max()))
        logger.debug("W_x: min value: {}; max value: {}".format(self.W_x.min(), self.W_x.max()))
        # self.h_tran has shape = (batch_size, seq_len, h_dim)
        # Since it is 3D, just pick one example to use for visualization:
        example_index = 0
        h_tran = self.h_tran[example_index, :, :]
        if h_tran is None:
            return
        logger.debug("H: min value: {}; max value: {}".format(h_tran.min(), h_tran.max()))
        h_sparsity = hoyer_sparsity(h_tran, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        logger.debug(f"H: sparsity: {mean_sparsity}")
        H = h_tran.t()
        if do_plots:
            plot_image_matrix(self.W_y, file_name=f"{self.config.results_dir}/{description}_W_y.png", 
                                title=f"{description}: W_y")
            plot_image_matrix(self.W_h, file_name=f"{self.config.results_dir}/{description}_W_h.png", 
                                title=f"{description}: W_h")
            plot_image_matrix(self.W_x, file_name=f"{self.config.results_dir}/{description}_W_x.png", 
                                title=f"{description}: W_x")
            plot_image_matrix(H, file_name=f"{self.config.results_dir}/{description}_H.png", 
                                title=f"{description}: H")
            W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
            V_pred = W @ H
            plot_image_matrix(V_pred, file_name=f"{self.config.results_dir}/{description}_V_pred.png", 
                                title=f"{description}: V_pred")
            y_pred = self.W_y @ H
            plot_image_matrix(y_pred, f"{self.config.results_dir}/{description}_y_pred.png", 
                title=f"{description}: Predicted values for training targets")
            
            h_prev_pred = self.W_h @ H
            plot_image_matrix(h_prev_pred, f"{self.config.results_dir}/{description}_h_prev_pred.png", 
                title=f"{description}: Predicted values for H_prev hidden states")
            
            x_pred = self.W_x @ H
            plot_image_matrix(x_pred, f"{self.config.results_dir}/{description}_x_pred.png", 
                title=f"{description}: Predicted values for X inputs")

    @torch.no_grad()
    def update_weights(self):
        """Perform the NMF learning update on W.

        The weights are updated by updating W in the following factorization:

        V approx= W * H

        using the masking matrix M which is the same shape as V. M has 0s corresponding
        to the outputs Y which are not used to compute the prediction loss and 1s elsewhere.

        Since we are using mini-batched factorization, we start with 3-dim matrices V_3d,
        H_3d, and M_3d. These need to be reshaped into 2d matrices.
        """
        
        W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
        # Prepare X:
        # shape = (batch_size, seq_len, h_dim)
        H_tran = self.h_tran
        assert H_tran is not None, "You must call forward() before update_weights()"
        # So we need to reshape it into a 2-dim matrix to call the following function.
        H_tran_2D = rearrange(H_tran, 'b s h -> (b s) h')
        if self.config.learning_rate is None:
            w_lr = fista_compute_lr_from_weights(H_tran_2D)*self.config.w_lr_scale
        else:
            w_lr = self.config.learning_rate

        # Now let's construct V.
        #
        #      Y
        # V =  H_prev
        #      X

        # shape: (batch_size, y_dim, seq_len)
        Y = self.y_targets
        # shape: (batch_size, h_dim, seq_len)
        H_prev = self.h_prev_targets
        # shape: (batch_size, x_dim, seq_len)
        X = self.X_targets

        V = torch.cat((Y, H_prev, X), dim = 1)
        # Reshape V into a 2-dim matrix.
        V_2D = rearrange(V, 'b d s -> d (b s)')

        # Masking matrix with shape: (batch_size, y_dim, seq_len)
        M_Y = self.M_Y
        # Masking matrix with shape: (batch_size, z_dim, seq_len)
        M_Z = self.M_Z

        # Masking matrix with same shape as V.
        M_V = torch.cat((M_Y, M_Z), dim = 1)
        # Reshape into a 2-dim matrix.
        M_V_2D = rearrange(M_V, 'b d s -> d (b s)')

        rows_W_y = self.config.y_dim
        rows_W_h = self.config.basis_vector_count
        row_start_W_h = rows_W_y + rows_W_h

        W = normalize_columns_at_most_equal_max_val(W, H_tran_2D)
        W = left_update_nmf_sgd(V_2D, 
                                W,
                                H_tran_2D,
                                learn_rate=w_lr,
                                shrinkage_sparsity=self.config.sparsity_L1_W,
                                weight_decay=self.config.weight_decay_W,
                                min_val=self.min_val,
                                force_nonnegative=self.config.enforce_nonneg_params,
                                M=M_V_2D,
                                )
        if False:
            # Apply scaling normalization to prevent W and/or H for exploding
            (W, H_tran_2D) = normalize_columns_equal_max_val(W, H_tran_2D)
        self.W_y = W[:rows_W_y, :]
        self.W_h = W[rows_W_y:row_start_W_h, :]
        self.W_x = W[row_start_W_h:, :]
        

    @torch.no_grad()
    def forward(self, X, targets = None):
        """Forward update.

        Compute the forward update on the RNN factorizations and return
        the loss.

        Args:
            x (tensor): float tensor of shape (batch_size, x_dim, seq_len)
            targets: (None or Tensor):  float tensor of shape (batch_size, y_dim, seq_len). This is the
                target sequence for y_t.
            h_prev: (tensor): tensor of (batch_size, size hidden_dim)

        Returns:
            y_pred (tensor): float tensor of shape (batch_size, y_dim, seq_len)
            loss (tensor): tensor of size 1. This is only returned if `targets` is not None.


        """
        (batch_size, x_dim, seq_len) = X.size()
        #H_prev = torch.zeros(batch_size, self.config.basis_vector_count, seq_len, device=X.device)
        H_prev = torch.rand(batch_size, self.config.basis_vector_count, seq_len, device=X.device)*self.config.weights_noise_scale
        assert H_prev.size() == (batch_size, self.config.basis_vector_count, seq_len)
        # Shape = (batch_size, z_dim, seq_len)
        Z = torch.cat((H_prev, X), dim=1)
        assert Z.size() == (batch_size, self.config.basis_vector_count + x_dim, seq_len)
        h_tran = torch.rand(batch_size, seq_len, self.config.basis_vector_count, device=X.device)*self.inference_noise_scale
        M_Y = torch.zeros_like(targets) # masking matrix for the targets
        M_Z = torch.ones_like(Z) # masking matrix for Z

        # weights
        W_z = torch.cat((self.W_h, self.W_x), dim=0)
        assert W_z.size() == (self.config.basis_vector_count + x_dim, self.config.basis_vector_count)
        # Factorization: Z approx= W_z * H
        if self.config.nmf_inference_algorithm == "sequential_sgd":
            # Iterate to convergence on each time slice, then use the converged state outptut
            # as the input for the next time slice.
            lr = fista_compute_lr_from_weights(W_z)*self.config.h_lr_scale
            y_pred_list = list()
            loss = 0
            # let's set the maximum allowable value for hidden states to be equal to the largest value
            # in X or W_z:
            max_allowed_value = X.max()
            per_slice_correct_predictions_cur_time_step = 0
            total_possible_predictions = 0
            start_slice = seq_len - self.copy_seq_len
            for t in range(seq_len):
                h_prev_t_slice =  H_prev[:, :, t]
                x_t_slice = X[:, :, t]
                z_t_slice = torch.cat((h_prev_t_slice, x_t_slice), dim=1)
                Z_temp = z_t_slice.t()
                H_tran_temp = h_tran[:, t, :]
                total_iters = self.config.nmf_inference_iters_no_grad
                for i in range(total_iters):
                    H_tran_temp = right_update_nmf_sgd(
                        Z_temp,
                        W_z,
                        H_tran_temp,
                        learn_rate=lr,
                        shrinkage_sparsity=self.config.sparsity_L1_H,
                        weight_decay = self.config.weight_decay_H,
                        force_nonnegative_H=True,
                    )
                    if False:
                        maxh = H_tran_temp.max()
                        if maxh > max_allowed_value:
                            # Normalize hidden state to its value does not exceed maximum input value in X.
                            H_tran_temp *= (max_allowed_value / maxh)

                    max_allowed_value = Z_temp.max()
                    maxh = H_tran_temp.max()
                    if maxh > max_allowed_value:
                        # Normalize hidden state to its value does not exceed maximum input value in X.
                        H_tran_temp = H_tran_temp * (max_allowed_value / maxh)

                    
                
                # Add the current inference result to h_tran:
                h_tran[:, t, :] = H_tran_temp[:, :]
                if t < seq_len - 1:
                    # Shape of H_tran_temp = (batch_size, h_dim)
                    H_prev[:, :, t+1] = H_tran_temp[:, :]

                # Predictions based on inferred H:
                # shape: (batch_size, y_dim)
                y_pred_t_slice = torch.einsum(
                        "ij,kj->ki", self.W_y, H_tran_temp)
                y_pred_list.append(y_pred_t_slice)
                # shape: (batch_size, h_dim)
                h_prev_pred_t_slice = torch.einsum(
                        "ij,kj->ki", self.W_h, H_tran_temp)
                # shape: (batch_size, x_dim)
                x_pred_t_slice = torch.einsum(
                        "ij,kj->ki", self.W_x, H_tran_temp)

                if t >= start_slice:                
                    # Update loss
                    # targets has shape: (batch_size, y_dim, seq_len)

                    # Y pred vs targets:
                    y_target_t_slice = targets[:, :, t]
                    loss_y = torch.nn.functional.mse_loss(y_pred_t_slice, y_target_t_slice, reduction='sum')/torch.numel(y_target_t_slice)

                    # H pred vs targets:
                    loss_h = torch.nn.functional.mse_loss(h_prev_pred_t_slice, h_prev_t_slice, reduction='sum')/torch.numel(h_prev_t_slice)

                    # (optional) X pred vs targets:
                    loss_x = torch.nn.functional.mse_loss(x_pred_t_slice, x_t_slice, reduction='sum')/torch.numel(x_t_slice)
                    loss = loss + loss_y + loss_h + loss_x

                    # todo: Also consider computing the full sequence accuracy, so that a single error makes the sequence predictions wrong.
                    # compute accuracy (per time slice):
                    pred_int = torch.argmax(y_pred_t_slice, dim=1)
                    targets_int = torch.argmax(y_target_t_slice, dim=1)
                    per_slice_correct_predictions_cur_time_step += pred_int.eq(targets_int).sum().item()
                    total_possible_predictions += torch.numel(pred_int)

                    # Update masking matrix
                    M_Y[:, :, t] = 1.0
                    
            # Save a copies for viewing stats and updating weights.
            # shape:  (batch_size, seq_len, h_dim)
            self.h_tran = h_tran
            loss = loss/seq_len
            y_pred = torch.stack(y_pred_list, dim=2)
            # Save targets:
            self.y_targets = targets
            self.h_prev_targets = H_prev
            self.X_targets = X
            self.Z_targets = Z
            self.M_Y = M_Y
            self.M_Z = M_Z
            return y_pred, loss, per_slice_correct_predictions_cur_time_step, total_possible_predictions
        else:
            raise ValueError(f"Bad value for nmf_inference_algorithm: {self.nmf_inference_algorithm}")



class FactorizedRNNWithoutBackpropSeqMNIST:
    """Factorized RNN for the Sequential MNIST task with conventional NMF learning and inference updates (no backprop).

    This is the factorized vanilla RNN as described in the paper.
    This version uses conventional NMF updates (i.e., without backpropagation). We
    iterate right (inference) updates until convergence and then perform a left 
    (learning) udpate.

    The image pixels are fed into the RNN a colummn of pixels at a time. 
    The RNN then makes the class prediction in the final
    time slice. E.g., in the MNIST column-wise classification task, there are 28 columns of
    28 pixels per column. So the model's output predictions are ignored for the first 27
    time slices and the loss is only computed on the (final) 28th time slice.
    Note that this means that all examples in the batch will be 28 time slices long.

    This version supports mini-batched processing. We simply extend the non-batched version to
    support batched operation by appending a batch dimension to X and Y.
    
    Input sequence of feature vectors: [x_0  x_1 x_2 ... x_T-1]
    Hidden state vectors: [h_-1 h_0 h_1 ... h_T-1]
    Target output sequence of vectors: [y_0  y_1 y_2 ... y_T-1]

    General factorization:

    [Y]      = [y_0  y_1 y_2 ... y_T-1]
    [H_prev] = [h_-1 h_0 h_1 ... h_T-2] approx= W [h_0 h_1 h_2 ... h_T-1]
    [X]      = [x_0  x_1 x_2 ... x_T-1]

        [W_y]
    W = [W_h]
        [W_x]

    W has size (x_dim + hidden_state_dim + y_dim, basis_vector_count).

    where hidden_state_dim = basis_vector_count, making W_h a square matrix.

    Notes:
    For the case of an autoregressive model (i.e., predict the next feature
    vector x_t), Y and W_y can be removed from the above factorization, leaving:

    Autoregressive factorization:

    [H_prev] = [h_-1 h_0 h_1 ... h_T-2] approx= W [h_0 h_1 h_2 ... h_T-1]
    [X]      = [x_0  x_1 x_2 ... x_T-1]

    W = [W_h]
        [W_x]
        
    For more details, see paper.

    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        if config.data_representaiton == "pixels":
            # pixel-wise
            self.x_dim = 1
        elif config.data_representaiton == "rows": # actually columns
            self.x_dim = config.image_side_pixels
        else:
            raise ValueError('oops')
        if config.enforce_nonneg_params:
            self.W_y = torch.rand(config.y_dim, config.basis_vector_count)*config.weights_noise_scale
            self.W_h = torch.rand(config.basis_vector_count, config.basis_vector_count)*config.weights_noise_scale
            self.W_x = torch.rand(self.x_dim, config.basis_vector_count)*config.weights_noise_scale
        else:
            self.W_y = (torch.rand(config.y_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_h = (torch.rand(config.basis_vector_count, config.basis_vector_count) - 0.5)*config.weights_noise_scale
            self.W_x = (torch.rand(self.x_dim, config.basis_vector_count) - 0.5)*config.weights_noise_scale
        self.W_y = self.W_y.to(device)
        self.W_h = self.W_h.to(device)
        self.W_x = self.W_x.to(device)
        self.h_tran = None
        self.inference_noise_scale = 0
        self.h_lr = 0
        self.w_lr = 0

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        """Print debug info.

        """
        print(f"Model info for: {description}")
        if self.config.nmf_inference_algorithm == "sgd":
            if isinstance(self.inference_sgd_lr, torch.Tensor):
                print(f"sgd learning rate: {self.inference_sgd_lr.item()}")
        print("W_y: min value: {}; max value: {}".format(self.W_y.min(), self.W_y.max()))
        print("W_h: min value: {}; max value: {}".format(self.W_h.min(), self.W_h.max()))
        print("W_x: min value: {}; max value: {}".format(self.W_x.min(), self.W_x.max()))
        print(f"H lr: {self.h_lr}")
        print(f"W lr: {self.w_lr}")
        # self.h_tran has shape = (batch_size, seq_len, h_dim)
        # Since it is 3D, just pick one example to use for visualization:
        example_index = 4
        h_tran = self.h_tran[example_index, :, :]
        if h_tran is None:
            return
        print("H: min value: {}; max value: {}".format(h_tran.min(), h_tran.max()))
        h_sparsity = hoyer_sparsity(h_tran, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        print("H: sparsity: ", mean_sparsity)
        H = h_tran.t()
        if do_plots:
            plot_image_matrix(self.W_y, file_name=f"{self.config.results_dir}/{description}_W_y.png", 
                                title=f"{description}: W_y")
            plot_image_matrix(self.W_h, file_name=f"{self.config.results_dir}/{description}_W_h.png", 
                                title=f"{description}: W_h")
            plot_image_matrix(self.W_x, file_name=f"{self.config.results_dir}/{description}_W_x.png", 
                                title=f"{description}: W_x")
            plot_image_matrix(H, file_name=f"{self.config.results_dir}/{description}_H.png", 
                                title=f"{description}: H")
            W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
            V_pred = W @ H
            plot_image_matrix(V_pred, file_name=f"{self.config.results_dir}/{description}_V_pred.png", 
                                title=f"{description}: V_pred")
            #y_pred = self.W_y @ H
            y_pred_3d = self.y_pred
            y_pred = y_pred_3d[example_index, :, :]
            plot_image_matrix(y_pred, f"{self.config.results_dir}/{description}_y_pred.png", 
                title=f"{description}: Predicted values for training targets")
            
            h_prev_pred = self.W_h @ H
            plot_image_matrix(h_prev_pred, f"{self.config.results_dir}/{description}_h_prev_pred.png", 
                title=f"{description}: Predicted values for H_prev hidden states")
            
            x_pred = self.W_x @ H
            plot_image_matrix(x_pred, f"{self.config.results_dir}/{description}_x_pred.png", 
                title=f"{description}: Predicted values for X inputs")
            
            # Now plot the target values:

            # Plot the Y (class label) targets:
            # shape: (batch_size, y_dim, seq_len)
            Y_targets = self.Y_targets_one_hot_all_slices
            # Extract just one example:
            Y_targets_1_example = Y_targets[example_index, :, :]
            plot_image_matrix(Y_targets_1_example, f"{self.config.results_dir}/{description}_Y_targets_1_example.png", 
                title=f"{description}: Y (class label) targets")

            # shape: (batch_size, h_dim, seq_len)
            H_prev_targets = self.h_prev_targets
            # Extract just one example:
            H_prev_targets_1_example = H_prev_targets[example_index, :, :]
            plot_image_matrix(H_prev_targets_1_example, f"{self.config.results_dir}/{description}_H_prev_targets_1_example.png", 
                title=f"{description}: H_prev (previous hidden states) targets")

            # shape: (batch_size, x_dim, seq_len)
            X_targets = self.X_targets
            # Extract just one example:
            X_targets_1_example = X_targets[example_index, :, :]
            plot_image_matrix(X_targets_1_example, f"{self.config.results_dir}/{description}_X_targets_1_example.png", 
                title=f"{description}: X (input features) targets")

            M_Y = self.M_Y
            M_Y_1_example = M_Y[example_index, :, :]
            plot_image_matrix(M_Y_1_example, f"{self.config.results_dir}/{description}_M_Y_1_example.png", 
                title=f"{description}: M_Y_1_example (mask for Y) targets")


    @torch.no_grad()
    def update_weights(self):
        """Perform the learning update.

        The weights are updated by updating W in the following factorization:

        V approx= W * H

        using the masking matrix M which is the same shape as V. M has 0s corresponding
        to the outputs Y which are not used to compute the prediction loss and 1s elsewhere.

        Since we are using mini-batched factorization, we start with 3-dim matrices V_3d,
        H_3d, and M_3d. These need to be reshaped into 2d matrices.
        """
        W = torch.cat((self.W_y, self.W_h, self.W_x), dim=0)
        # Prepare X:
        # shape = (batch_size, seq_len, h_dim)
        H_tran = self.h_tran
        assert H_tran is not None, "You must call forward() before update_weights()"
        # So we need to reshape it into a 2-dim matrix to call the following function.
        H_tran_2D = rearrange(H_tran, 'b s h -> (b s) h')
        if self.config.learning_rate is None:
            w_lr = fista_compute_lr_from_weights(H_tran_2D)*self.config.w_lr_scale
        else:
            w_lr = self.config.learning_rate

        # log lr:
        self.w_lr = w_lr

        # Now let's construct V.
        #
        #      Y
        # V =  H_prev
        #      X

        # shape: (batch_size, y_dim, seq_len)
        Y = self.Y_targets_one_hot_all_slices
        # shape: (batch_size, h_dim, seq_len)
        H_prev = self.h_prev_targets
        # shape: (batch_size, x_dim, seq_len)
        X = self.X_targets

        V = torch.cat((Y, H_prev, X), dim = 1)
        # Reshape V into a 2-dim matrix.
        V_2D = rearrange(V, 'b d s -> d (b s)')

        # Masking matrix with shape: (batch_size, y_dim, seq_len)
        M_Y = self.M_Y
        # Masking matrix with shape: (batch_size, z_dim, seq_len)
        M_Z = self.M_Z

        # Masking matrix with same shape as V.
        M_V = torch.cat((M_Y, M_Z), dim = 1)
        # Reshape into a 2-dim matrix.
        M_V_2D = rearrange(M_V, 'b d s -> d (b s)')

        rows_W_y = self.config.y_dim
        rows_W_h = self.config.basis_vector_count
        row_start_W_h = rows_W_y + rows_W_h
        
        W = normalize_columns_at_most_equal_max_val(W, H_tran_2D)
        W = left_update_nmf_sgd(V_2D, 
                                    W,
                                    H_tran_2D,
                                    learn_rate=w_lr,
                                    shrinkage_sparsity=self.config.sparsity_L1_W,
                                    weight_decay=self.config.weight_decay_W,
                                    min_val=1e-5,
                                    force_nonnegative=self.config.enforce_nonneg_params,
                                    M=M_V_2D)
        #if False:
        #    (W, H_tran_2D) = normalize_columns_equal_max_val(W, H_tran_2D)
        self.W_y = W[:rows_W_y, :]
        self.W_h = W[rows_W_y:row_start_W_h, :]
        self.W_x = W[row_start_W_h:, :]
        

    @torch.no_grad()
    def forward(self, x, targets = None):
        """Forward update.

        Compute the forward update on the RNN factorizations and return
        the loss.

        Args:
            x (tensor): float tensor of shape (batch_size, x_dim, seq_len)
            targets: (None or Tensor):  float tensor of shape (batch_size, y_dim, seq_len). This is the
                target sequence for y_t.
            h_prev: (tensor): tensor of (batch_size, size hidden_dim)

        Returns:
            y_pred (tensor): float tensor of shape (batch_size, y_dim, seq_len)
            loss (tensor): tensor of size 1. This is only returned if `targets` is not None.
            correct_predictions_cur_time_step
            total_possible_predictions


        """
        # data has shape (batch_size, 1, height, width).
        if self.config.data_representaiton == "pixels":
            # pixel-wise
            x = rearrange(x, 'b 1 h w -> b 1 (h w)')
        elif self.config.data_representaiton == "rows":
            x = rearrange(x, 'b c h w -> b h (c w)')
        elif self.config.data_representaiton == "exp":
            x = rearrange(x, 'b 1 h w -> b 1 (h w)')
            x = x.reshape((x.shape[0], x.shape[1]*self.x_dim, -1))
        else:
            raise ValueError('oops')
        # x_train has shape (batch_size, height*width)
        (batch_size, x_dim, seq_len) = x.size()
        
        #H_prev = torch.zeros(batch_size, self.config.basis_vector_count, seq_len, device=x.device) # also works
        H_prev = torch.rand(batch_size, self.config.basis_vector_count, seq_len, device=x.device)*self.config.weights_noise_scale
        assert H_prev.size() == (batch_size, self.config.basis_vector_count, seq_len)
        # Shape = (batch_size, z_dim, seq_len)
        Z = torch.cat((H_prev, x), dim=1)
        assert Z.size() == (batch_size, self.config.basis_vector_count + x_dim, seq_len)
        h_tran = torch.rand(batch_size, seq_len, self.config.basis_vector_count, device=x.device)*self.inference_noise_scale

        # masking matrix for the targets
        Y_targets_one_hot_all_slices = torch.zeros((batch_size, self.config.y_dim, seq_len), device=x.device)
        M_Y = torch.zeros((batch_size, self.config.y_dim, seq_len), device=x.device)
        M_Z = torch.ones_like(Z)*self.config.z_mask_val # masking matrix for Z

        # weights
        W_z = torch.cat((self.W_h, self.W_x), dim=0)
        assert W_z.size() == (self.config.basis_vector_count + x_dim, self.config.basis_vector_count)
        # Factorization: Z approx= W_z * H
        # Iterate to convergence on each time slice, then use the converged state outptut
        # as the input for the next time slice.
        if self.config.learning_rate_H is None:
            lr = fista_compute_lr_from_weights(W_z)*self.config.h_lr_scale
        else:
            lr = self.config.learning_rate_H

        self.h_lr = lr
        y_pred_list = list()
        # let's set the maximum allowable value for hidden states to be equal to the largest value
        # in X or W_z:
        max_allowed_value = x.max()
        correct_predictions_cur_time_step = 0
        total_possible_predictions = 0
            
        for t in range(seq_len):      
            h_prev_t_slice =  H_prev[:, :, t]
            x_t_slice = x[:, :, t]        
            z_t_slice = torch.cat((h_prev_t_slice, x_t_slice), dim=1)
            Z_temp = z_t_slice.t()
            H_tran_temp = h_tran[:, t, :]
            
            if False:
                total_iters = self.config.nmf_inference_iters_no_grad
                tolerance = self.config.fista_tolerance
                H_tran_temp = fista_right_update(Z_temp, W_z, H_tran_temp, tolerance=tolerance, 
                                            max_iterations=total_iters,
                                            shrinkage_sparsity=self.config.sparsity_L1_H,
                                            debug=False)

                if True:
                    # This is faster since to wait to normalize until after convergence of the current time slice of H.
                    maxh = H_tran_temp.max()
                    if maxh > max_allowed_value:
                        # Normalize hidden state to its value does not exceed maximum input value in X.
                        H_tran_temp *= (max_allowed_value / maxh)
            if True:
                for i in range(self.config.nmf_inference_iters_no_grad):
                    H_tran_temp = right_update_nmf_sgd(
                            Z_temp,
                            W_z,
                            H_tran_temp,
                            learn_rate=lr,
                            shrinkage_sparsity=self.config.sparsity_L1_H,
                            weight_decay=self.config.weight_decay_H,
                            force_nonnegative_H=True)
                                
                    max_allowed_value = Z_temp.max()
                    maxh = H_tran_temp.max()
                    if maxh > max_allowed_value:
                        # Normalize hidden state to its value does not exceed maximum input value in X.
                        H_tran_temp = H_tran_temp * (max_allowed_value / maxh)
    
            # Add the current inference result to h_tran:
            h_tran[:, t, :] = H_tran_temp[:, :]
            if t < seq_len - 1:
                # Shape of H_tran_temp = (batch_size, h_dim)                    
                # (batch_size, h_dim, seq_len)
                H_prev[:, :, t+1] = H_tran_temp[:, :]

            # Predictions based on inferred H:
            # shape: (batch_size, y_dim)
            y_pred_t_slice = torch.einsum(
                    "ij,kj->ki", self.W_y, H_tran_temp)
            y_pred_list.append(y_pred_t_slice)
            # shape: (batch_size, h_dim)
            h_prev_pred_t_slice = torch.einsum(
                    "ij,kj->ki", self.W_h, H_tran_temp)
            # shape: (batch_size, x_dim)
            x_pred_t_slice = torch.einsum(
                    "ij,kj->ki", self.W_x, H_tran_temp)

            if t == seq_len - 1:
                # Update loss
                # targets has shape: (batch_size,)
                # Y pred vs targets:
                y_target_t_slice = targets
                y_targets_one_hot = torch.zeros((batch_size, self.config.y_dim), device=x.device)
                all_ones = torch.ones([batch_size, 1], dtype=torch.float, device = x.device)
                y_targets_one_hot.scatter_(1, targets[:, None], all_ones)
                Y_targets_one_hot_all_slices[:, :, t] = y_targets_one_hot[:, :]
                loss_y = torch.nn.functional.mse_loss(y_pred_t_slice, y_targets_one_hot, reduction='sum')/torch.numel(y_target_t_slice)
                # H pred vs targets:
                loss_h = torch.nn.functional.mse_loss(h_prev_pred_t_slice, h_prev_t_slice, reduction='sum')/torch.numel(h_prev_t_slice)
                # (optional) X pred vs targets:
                loss_x = torch.nn.functional.mse_loss(x_pred_t_slice, x_t_slice, reduction='sum')/torch.numel(x_t_slice)
                loss = loss_y + loss_h + loss_x
                # compute accuracy:
                pred_int = torch.argmax(y_pred_t_slice, dim=1)
                correct_predictions_cur_time_step += pred_int.eq(targets).sum().item()
                total_possible_predictions += torch.numel(pred_int)
                # Update masking matrix
                M_Y[:, :, t] = 1.0

        # Save a copies for viewing stats and updating weights.
        # shape:  (batch_size, seq_len, h_dim)
        self.h_tran = h_tran
        loss = loss/seq_len
        y_pred = torch.stack(y_pred_list, dim=2)
        self.y_pred = y_pred
        # Save targets:
        self.y_targets = targets
        self.h_prev_targets = H_prev
        self.X_targets = x
        self.Z_targets = Z
        self.M_Y = M_Y
        self.M_Z = M_Z
        self.Y_targets_one_hot_all_slices = Y_targets_one_hot_all_slices
        return y_pred, loss, correct_predictions_cur_time_step, total_possible_predictions

