import random
import sys
import random
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from utils.utils import ValueAccumulator
from utils.utils import *
import os
from run_experiments import VanillaRNN, FactorizedRNN
import musdb
import fast_bss_eval


class BasicRNNEvaluator:
    """Evaluate the forward pass of an RNN.
    
    This is intended to be used for tasks in which the loss is computed for
    all time slices.
    This is compatible with both conventional and factorized RNNs.
    
    """

    def __init__(self, model, device) -> None:
        self.model = model
        self.h_dim = self.model.hidden_dim
        self.has_cell = model.has_cell
        # previous hidden state
        self.h_prev = None
        if self.has_cell:
            # previous cell state (if LSTM)
            self.c_prev = None
        self.device = device

    @torch.no_grad()
    def print_stats(self, do_plots=True, description=""):
        """Print debug info.

        """
        print(f"Model info for: {description}")
        self.model.print_stats(do_plots=do_plots, description=description)
        # self.h_tran has shape = (batch_size, seq_len, h_dim)
        # Since it is 3D, just pick one example to use for visualization:
        example_index = 0
        # h_prev shape = (h_dim, seq_len)
        h_prev = self.h_prev_all_slices[example_index, :, :]
        # h shape = (h_dim, seq_len)
        h = self.h_all_slices[example_index, :, :]
        # x shape = (x_dim, seq_len)
        x = self.x_targets_all_slices[example_index, :, :]
        print("H: min value: {}; max value: {}".format(h.min(), h.max()))
        h_sparsity = hoyer_sparsity(h, dim=1)
        mean_sparsity = torch.mean(h_sparsity).item()
        print("H: sparsity: ", mean_sparsity)
        
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
                print("Current RNN not supported for plots")
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
            y_1hot_targets = self.y_targets_all_slices[example_index, :, :] 
            plot_image_matrix(y_1hot_targets, f"{config.results_dir}/{description}_Y_targets_1_example.png", 
                title=f"{description}: Y (class label) targets")

            plot_image_matrix(h_prev, f"{config.results_dir}/{description}_H_prev_targets_1_example.png", 
                title=f"{description}: H_prev (previous hidden states) targets")

            plot_image_matrix(x, f"{config.results_dir}/{description}_X_targets_1_example.png", 
                title=f"{description}: X (input features) targets")


    def forward(
        self,
        x_train,
        y_targets,
        is_reset = None,
        enable_backprop_through_time = True,
        loss_type="mse",
    ):
        """Evaluate the on a batch of inputs.

        Batches should be supplied so that for each example in the batch, the first token/feature
        in the sequence immediately follows the last token of the previous batch.

        Args:
            x_train (Tensor of float): Input features of shape (batch_size, feature_dim, seq_len).
            y_targets (Tensor of float or int): Target (ground truth) features of shape (batch_size, feature_dim, seq_len).
            is_reset (Tensor of boolean or None). Shape is (batch_size,) and is True if the i'th example
                in the batch is the start of a new sequence, in which case the internal state
                for this example needs to be reset.
        """
        (batch_size, feature_dim, seq_len) = x_train.size()
        device = x_train.device
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

        if is_reset is None:
            is_reset = torch.ones((batch_size), dtype=bool, device=device)

        # Reset the internal state of any examples in the batch where is_reset is True.
        is_reset_h = torch.unsqueeze(is_reset, 1)
        is_reset_h = is_reset_h.expand(batch_size, self.h_prev.size(1))
        self.h_prev = torch.where(is_reset_h, h_zero, self.h_prev)
        if self.has_cell:
            is_reset_c = torch.unsqueeze(is_reset, 1)
            is_reset_c = is_reset_h.expand(batch_size, self.c_prev.size(1))
            self.c_prev = torch.where(is_reset_c, c_zero, self.c_prev)

        model = self.model
        # Create empty tensors for logging/visualization only:
        self.y_pred_all_slices = torch.zeros((batch_size, self.model.out_dim, seq_len), device=device)
        self.x_all_slices = torch.zeros((batch_size, feature_dim, seq_len), device=device)
        self.h_prev_all_slices = torch.zeros((batch_size, self.h_dim, seq_len), device=device)
        self.h_all_slices = torch.zeros((batch_size, self.h_dim, seq_len), device=device)

        with torch.no_grad():
            self.y_targets_all_slices = y_targets.clone().detach()
            self.x_targets_all_slices = x_train.clone().detach()

        loss_h = 0
        loss_x = 0
        loss_y = 0
        for m in range(seq_len):
            x = x_train[:, :, m] 
            if self.has_cell:
                y_t, c, h = model(x, self.c_prev, self.h_prev)
            else:
                y_t, h = model(x, self.h_prev)

            # logging
            with torch.no_grad():
                self.y_pred_all_slices[:, :, m] = y_t[:, :]
                self.h_prev_all_slices[:, :, m] = self.h_prev[:, :]
                self.h_all_slices[:, :, m] = h[:, :]

            cur_targets = y_targets[:, :, m]

            if loss_type == "cross_entropy":
                cur_targets_int = torch.argmax(cur_targets, dim=1)
                loss_y = loss_y + F.cross_entropy(y_t, cur_targets_int)
            elif loss_type == "mse":
                loss_y = loss_y + torch.nn.functional.mse_loss(y_t, cur_targets, reduction="sum")/torch.numel(cur_targets)
            elif loss_type == "mse_factorized":
                loss_y = loss_y +  torch.nn.functional.mse_loss(y_t, cur_targets, reduction="sum")/torch.numel(cur_targets)

                h_prev_pred = torch.einsum("ij,kj->ki", model.W_h, h)
                h_prev_targets = self.h_prev.clone().detach()
                loss_h = loss_h + torch.nn.functional.mse_loss(h_prev_pred, h_prev_targets, reduction='sum')/torch.numel(h_prev_targets)

                x_pred = torch.einsum("ij,kj->ki", model.W_x, h)
                x_targets = x.clone().detach()
                loss_x = loss_x + torch.nn.functional.mse_loss(x_pred, x_targets, reduction='sum')/torch.numel(x_targets)
            else:
                raise ValueError("Bad loss type")

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

        # Note: Just like a regular RNN, we only do 1 optimizer update per batch.
        if loss_type == "mse":
            loss = loss_y / seq_len
        elif loss_type == "cross_entropy":
            loss = loss_y / seq_len
        elif loss_type == "mse_factorized":
            loss = (loss_y + loss_h + loss_x)/seq_len
        return self.y_pred_all_slices, loss


class FastMusDBDatasetLoader:
    """MUSDB18 dataset loader to use for validation and test dataset splits.
    
    The existing MusDB18 dataset loaders seem super slow, so I wrote this to use instead.

    This dataset loader selects a batch of random track locations in the specified dataset
    (train, validation, or test) and creates batches of (input_batch, target_batch) containing
    the inputs (mixed) and targets (single audio source/stem).

    This dataset loader should be used for validation and test dataset splits since it constructs
    the input and output (i.e., target) examples in the usual expected way. That is, for each example,
    the input audio uses the same audio sample offsets for all audio sources (stems). The output
    audio then consists only of the specified target source (e.g., 'vocals') without any of the
    other sources.

    This is fine for evaluation, but is not ideal for training. For training, we might prefer
    to randomize the starting audio locations for all of the available sources to maximize
    the variability between training examples. There is another dataset loader that should
    be used instead for that case.

    Usage:

    Install musdb from: https://github.com/sigsep/sigsep-mus-db
    pip install musdb
    """

    def __init__(self, subsets, batch_size, batch_sample_count, split=None, target = 'vocals',  root="/home/vogel/datasets/musdb18hq"):
        """

        Usage:

        To get the training dataset:
        subsets = "train" and split = None

        To get the test dataset:
        subsets = "test" and split = None

        To get the "train" subset of the training dataset:
        subsets = "train" and split = "train"

        To get the "validation" subset of the training dataset:
        subsets = "train" and split = "valid"

        Args:
            subsets (str): Either "train" or "test"
            target (str): Target stem (instrument) track. E.g., 'vocals'
            root (str): Path to musdb18,
            batch_sample_count (int): Number of audio samples for example. E.g., 44100*5
            split (str or None): Optionally select the train or validation subset of the training dataset of not None.
        """

        mus_db = musdb.DB(root=root, is_wav=True, subsets=subsets, split=split)
        n = 0
        in_list = []
        out_list = []
        dataset_sample_count = 0
        print("Loading tracks...")
        for track in mus_db:
            in_audio = track.audio
            assert in_audio.ndim == 2
            in_audio = np.sum(in_audio, axis=1).astype(np.float32)*0.5 # convert to mono
            in_list.append(in_audio)
            out_audio = track.targets[target].audio
            assert out_audio.ndim == 2
            out_audio = np.sum(out_audio, axis=1).astype(np.float32)*0.5 # convert to mono
            out_list.append(out_audio)
            in_len = len(in_audio)
            dataset_sample_count += in_len
            print(f"track: {n} | num_samples: {in_len}")
            n += 1
        self.in_list = in_list
        self.out_list = out_list
        self.batch_size = batch_size
        self.batch_sample_count = int(batch_sample_count)
        print(f"Total audio samples in dataset: {dataset_sample_count}")
        self.dataset_sample_count = dataset_sample_count
        self.samples_per_batch = batch_size*batch_sample_count
        self.current_epoch_samples = 0
        print("Loading complete.")

    def reset(self):
        self.current_epoch_samples = 0

    def get_epochs(self):
        factional_epochs = self.current_epoch_samples/self.dataset_sample_count
        return factional_epochs

    def get_batch(self):
        in_list = self.in_list
        out_list = self.out_list
        assert len(in_list) == len(out_list), "Mismatched lengths of in_list and out_list"
    
        # Randomly select 'batch_size' track indices without replacement
        #selected_tracks = random.sample(range(len(in_list)), self.batch_size)
        # Randomly select 'batch_size' track indices with replacement
        selected_tracks = random.choices(range(len(in_list)), k=self.batch_size)

        batch_sample_count = self.batch_sample_count
        # Initialize empty arrays for storing the mini-batch
        x_batch = np.zeros((self.batch_size, batch_sample_count), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, batch_sample_count), dtype=np.float32)
    
        for i, track_idx in enumerate(selected_tracks):
            in_audio = in_list[track_idx]
            out_audio = out_list[track_idx]
        
            # Randomly select a starting point for this track
            start_idx = random.randint(0, len(in_audio) - batch_sample_count)
        
            # Extract the samples from start_idx to start_idx + batch_sample_count
            x_batch[i, :] = in_audio[start_idx:start_idx + batch_sample_count]
            y_batch[i, :] = out_audio[start_idx:start_idx + batch_sample_count]
    
        # update epoch info
        self.current_epoch_samples += self.samples_per_batch
        return x_batch, y_batch


class TrainingMusDBDatasetLoader:
    """MUSDB18 dataset loader to use for train dataset splits.
    
    The existing MusDB18 dataset loaders seem super slow, so I wrote this to use instead.

    This dataset loader selects a batch of random track locations in the specified dataset
    (train, validation, or test) and creates batches of (input_batch, target_batch) containing
    the inputs (mixed) and targets (single audio source/stem). Although any dataset split can
    be selected, this loader is designed mainly for loading examples during training, since
    the audio that is mixed to create the input examples is not aligned. This effectively
    provides for more variation in the training examples compared to a data loader that
    keeps the audio aligned. This loader can also optionally apply a random scaling to each
    source to further increase the variation in the training examples.

    MUSDB provides the following audio sources: ['mixture', 'drums', 'bass', 'other', 'vocals'].
    
    Usage:

    Install musdb from: https://github.com/sigsep/sigsep-mus-db
    pip install musdb
    """

    def __init__(self, subsets,
                 batch_size,
                 batch_sample_count,
                 split=None,
                 target = 'vocals',
                 root="/home/vogel/datasets/musdb18hq"):
        """

        Usage:

        To get the training dataset:
        subsets = "train" and split = None

        To get the test dataset:
        subsets = "test" and split = None

        To get the "train" subset of the training dataset:
        subsets = "train" and split = "train"

        To get the "validation" subset of the training dataset:
        subsets = "train" and split = "valid"

        Args:
            subsets (str): Either "train" or "test"
            sources (str): Audio sources to mix together (unaligned) to create the input audio.
            target (str): Target stem (instrument) track. E.g., 'vocals'
            root (str): Path to musdb18,
            batch_sample_count (int): Number of audio samples for example. E.g., 44100*5
            split (str or None): Optionally select the train or validation subset of the training dataset of not None.
        """

        mus_db = musdb.DB(root=root, is_wav=True, subsets=subsets, split=split)
        n = 0
        # Exclude the 'mixture' source since it is redundant.
        self.sources=['drums', 'bass', 'other', 'vocals']
        self.source_dict = {source: [] for source in self.sources}
        self.target = target
        out_list = []
        dataset_sample_count = 0
        print("Loading tracks...")
        #track_len = None
        for track in mus_db:
            for source in self.sources:
                in_audio = track.targets[source].audio
                assert in_audio.ndim == 2
                in_audio = np.sum(in_audio, axis=1).astype(np.float32)*0.5 # convert to mono
                self.source_dict[source].append(in_audio)
                track_len = len(in_audio)

            dataset_sample_count += track_len
            print(f"track: {n} | num_samples: {track_len}")
            n += 1

        self.out_list = out_list
        self.batch_size = batch_size
        self.batch_sample_count = int(batch_sample_count)
        print(f"Total audio samples in dataset: {dataset_sample_count}")
        self.dataset_sample_count = dataset_sample_count
        self.samples_per_batch = batch_size*batch_sample_count
        self.current_epoch_samples = 0
        print("Loading complete.")

    def reset(self):
        self.current_epoch_samples = 0

    def get_epochs(self):
        factional_epochs = self.current_epoch_samples/self.dataset_sample_count
        return factional_epochs

    def get_batch(self):
        batch_sample_count = self.batch_sample_count
        # Initialize empty arrays for storing the mini-batch
        x_batch = np.zeros((self.batch_size, batch_sample_count), dtype=np.float32)
        y_batch = np.zeros((self.batch_size, batch_sample_count), dtype=np.float32)

        # Scale each source by a random value
        low_limit = 0.5
        high_limit = 2.0
        array_length = len(self.sources)
        rand_array = np.random.uniform(low_limit, high_limit, array_length)
        source_scale_factors_dict = dict(zip(self.sources, rand_array))
        
        for source_name, audio_list in self.source_dict.items():
            # Randomly select 'batch_size' track indices with replacement
            selected_tracks = random.choices(range(len(audio_list)), k=self.batch_size)
            scale_factor = source_scale_factors_dict[source_name]
            for i, track_idx in enumerate(selected_tracks):
                audio = audio_list[track_idx]
                # Randomly select a starting point for this track
                start_idx = random.randint(0, len(audio) - batch_sample_count)
                # Extract the samples from start_idx to start_idx + batch_sample_count
                extracted_audio = audio[start_idx:start_idx + batch_sample_count]
                scaled_audio = scale_factor*extracted_audio
                x_batch[i, :] += scaled_audio
                if source_name == self.target:
                    y_batch[i, :] = scaled_audio
    
        # update epoch info
        self.current_epoch_samples += self.samples_per_batch
        return x_batch, y_batch


# Main config file for MUSDB source separation training and evaluation.
config_musdb18_rnn = AttributeDict(device = "cuda",
                           results_dir = "debug_plots",
                            sample_rate_hz=44100,
                            window_size = 2048,
                            hop_size = 1024,
                            fista_tolerance = None,
                            max_freq_bin_plot = 200,
                            rnn_type = "vanillaRNN",
                            #rnn_type = "FactorizedRNN",
                            hidden_dim = 1024,
                            audio_feature_type = "stft",
                            learning_rate = 1e-4, # Try 5e-4 for factorized RNN. Try 1e-4 for VanillaRNN
                            weight_decay = 5e-5,
                            batch_size = 100,
                            recurrent_drop_prob = 0.0,
                            input_drop_prob = 0.0,
                            enable_backprop_through_time = True,
                            enforce_nonneg_params = False,
                            nmf_inference_iters_no_grad = 0, # 0 # only used by factorized RNN
                            nmf_gradient_iters_with_grad = 10, # 10 only used by factorized RNN
                            learning_rate_H = None, # only used by factorized RNN
                            sparsity_L1_H = 0, # only used by factorized RNN
                            weight_decay_H = 0, # only used by factorized RNN
                            loss_type = "mse", # for both factorized and vanilla RNN
                            #loss_type = "mse_factorized", # only for factorized RNN
                            print_train_stats_every_iterations = 250,
                            evaluate_every_iterations = 250,
                            train_epoch_count = 300,
                            audio_clip_length_seconds = 5.0,
                            audio_sample_rate_hz = 44100,
                            root_compress_amount = 0.25,
                            feature_dim = 350, # limit feature dim to this (must be less than or equal to actual feature dim)
                            weights_noise_scale = 1e-2, #only used by factorized RNN
                            nmf_inference_algorithm = "fista", # works best, only used by factorized RNN
                            save_model_params_file = 'saved_models/network_parameters.pth',
                            musdb_root = "/home/vogel/datasets/musdb18hq",
                            train_loader = "TrainingMusDBDatasetLoader",
                            debug = False,                     
    )


@torch.no_grad()
def samples_to_spectrograms2(x, config):
    """Convert batch of audio samples to batch of spectrograms

    Args:
        x (tensor of float): shape = (batch_size, num_samples)

    Returns:
        (phase, scaled_compressed_mag_stft) (tensor of float): shape = (batch_size, num_freq_bins, num_time_slices)
            However, phase has the number of freq bins produced by torch.stft() while scaled_compressed_mag_stft
            may have fewer bins do to cropping.
    """
    fft_size = config.window_size
    hann_window = torch.hann_window(fft_size, dtype=x.dtype, device=x.device)
    spectrogram_complex = torch.stft(x, 
                                     n_fft=fft_size, 
                                     hop_length=config.hop_size, 
                                     win_length=fft_size,
                                     window=hann_window,
                                     return_complex=True)
    magnitude_stft = torch.abs(spectrogram_complex)
    (batch_size, freq_bins, time_bins) = magnitude_stft.size()
    # Normalize to keep phase information
    phase = spectrogram_complex / (magnitude_stft + 1e-9)
    max_magnitude = magnitude_stft.max()
    # Upper bound for magnitude.
    # It could be lower depending on window function, etc.
    max_possible_magnitude = fft_size/2.0
    assert max_possible_magnitude >= max_magnitude
    compressed_mag_stft = magnitude_stft**config.root_compress_amount
    max_possible_compressed_mag = max_possible_magnitude**config.root_compress_amount
    # Limit the maximum magnitude of the compressed and scaled magnitude part to at most 1.0.
    scaled_compressed_mag_stft = compressed_mag_stft/max_possible_compressed_mag
    assert scaled_compressed_mag_stft.max() <= 1.0
    # Optional: crop:
    if config.feature_dim is not None:
        assert config.feature_dim <= freq_bins, "config.feature_dim is too large."
        scaled_compressed_mag_stft = scaled_compressed_mag_stft[:, :config.feature_dim, :]
    return phase, scaled_compressed_mag_stft


@torch.no_grad()
def spectrograms_to_samples(phase, scaled_compressed_mag_stft, config):
    """Inverse of samples_to_spectrograms2.

    Given the phase (over all freq bins) and a possibly cropped and compressed
    magnitude spectrogram (possibly fewer freq bins), invert to audio signals.

    Returns:
        audio (tensor): Shape (batch_size, num_samples) audio signals in [-1, 1].
    """
    # Uncrop:
    full_freqs_mag = torch.zeros_like(phase)
    if config.feature_dim is not None:
        full_freqs_mag[:, :config.feature_dim, :] = scaled_compressed_mag_stft[:, :, :]
    else:
        full_freqs_mag = scaled_compressed_mag_stft

    # Now undo the compression and scaling.
    max_possible_magnitude = config.window_size/2.0
    max_possible_compressed_mag = max_possible_magnitude**config.root_compress_amount
    unscaled_magnitude_stft = (full_freqs_mag*max_possible_compressed_mag)**(1/config.root_compress_amount)    
    # Reconstruct spectrogram with original phase
    modified_spectrogram_complex = unscaled_magnitude_stft * phase
    # invert to audio
    hann_window = torch.hann_window(config.window_size, 
                                    dtype=scaled_compressed_mag_stft.dtype, 
                                    device=scaled_compressed_mag_stft.device)
    waveform_reconstructed = torch.istft(modified_spectrogram_complex, 
                                         n_fft=config.window_size, 
                                         hop_length=config.hop_size, 
                                         win_length=config.window_size,
                                         window=hann_window)
    return waveform_reconstructed


def train_musdb18_rnn():
    """Train a simple RNN-based source separation model on musdb18-hq.


    """
    config = config_musdb18_rnn
    # Add logger to the config.
    logger = configure_logger()
    config.logger = logger
    print(f"Using device: {config.device}")
    if config.rnn_type == "vanillaRNN":
        rnn = VanillaRNN(config, config.feature_dim, config.hidden_dim, config.feature_dim,
                         recurrent_drop_prob=config.recurrent_drop_prob,
                            input_drop_prob=config.input_drop_prob,
                            enable_bias = True,
                            enable_input_layernorm=False,
                            enable_state_layernorm=True)
    elif config.rnn_type == "FactorizedRNN":
        rnn = FactorizedRNN(config, config.feature_dim, config.hidden_dim, config.feature_dim)
    else:
        raise ValueError(f"Unrecognized value for config model_name: {config.rnn_type}")
    rnn = rnn.to(config.device)
    train_evaluator = BasicRNNEvaluator(rnn, device=config.device)
    optimizer = RMSpropOptimizerCustom(rnn.parameters(), default_lr=config.learning_rate)
    optimizer.weight_decay_hook(config.weight_decay)
    print(f"Using device: {config.device}")
    print("Loading training dataset")
    if config.train_loader == "TrainingMusDBDatasetLoader":
        train_loader = TrainingMusDBDatasetLoader(subsets="train", batch_size=config.batch_size,
                                          batch_sample_count=config.audio_clip_length_seconds*config.audio_sample_rate_hz,
                                          split="train",
                                          target="vocals",
                                          root=config.musdb_root)
    elif config.train_loader == "FastMusDBDatasetLoader":
        train_loader = FastMusDBDatasetLoader(subsets="train", batch_size=config.batch_size,
                                          batch_sample_count=config.audio_clip_length_seconds*config.audio_sample_rate_hz,
                                          split="train",
                                          target="vocals",
                                          root=config.musdb_root)
    else:
        raise ValueError("bad value")
    print("Loading validation dataset")
    validation_loader = FastMusDBDatasetLoader(subsets="train", batch_size=config.batch_size,
                                          batch_sample_count=config.audio_clip_length_seconds*config.audio_sample_rate_hz,
                                          split="valid",
                                          target="vocals",
                                          root=config.musdb_root)

    train_loss_accumulator = ValueAccumulator()
    train_loss_list = []
    validation_loss_list = []
    train_epoch_list = []
    n = 0
    best_validation_loss = None
    best_validation_epoch = 0
    while train_loader.get_epochs() < config.train_epoch_count:
        (x, y) = train_loader.get_batch()
        x = np2t(x)
        y = np2t(y)
        x = x.to(config.device)
        y = y.to(config.device)
        # x an y have shape = (batch_size, num_channels, num_samples)
        with torch.no_grad():
            (_, x_specgram) = samples_to_spectrograms2(x, config)
            (_, y_specgram) = samples_to_spectrograms2(y, config)
        
        y_specgram_pred, loss = train_evaluator.forward(
                x_specgram,
                y_specgram,
                enable_backprop_through_time=config.enable_backprop_through_time,
                loss_type=config.loss_type,
        )
        train_loss_accumulator.accumulate(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if config.enforce_nonneg_params:
            rnn.clip_weights_nonnegative()
        sys.stdout.write(".")
        sys.stdout.flush()
        if n % config.evaluate_every_iterations == 0 and n > 0:
            print()
            print("Evaluating model on the validation dataset...")
            rnn.train(False)
            validation_evaluator = BasicRNNEvaluator(rnn, device=config.device)
            validation_loss_accumulator = ValueAccumulator()
            while validation_loader.get_epochs() < 1:
                (x, y) = validation_loader.get_batch()
                x = np2t(x)
                y = np2t(y)
                x = x.to(config.device)
                y = y.to(config.device)
                # x an y have shape = (batch_size, num_channels, num_samples)
                with torch.no_grad():
                    (_, x_specgram) = samples_to_spectrograms2(x, config)
                    (_, y_specgram) = samples_to_spectrograms2(y, config)
        
                    y_specgram_pred, loss = validation_evaluator.forward(
                            x_specgram,
                            y_specgram,
                            enable_backprop_through_time=config.enable_backprop_through_time,
                            loss_type=config.loss_type,
                    )
                validation_loss_accumulator.accumulate(loss.item())
                sys.stdout.write(".")
                sys.stdout.flush()
            train_loss = train_loss_accumulator.get_mean()
            train_loss_list.append(train_loss)
            validation_loss = validation_loss_accumulator.get_mean()
            validation_loss_list.append(validation_loss)
            train_epoch_list.append(train_loader.get_epochs())
            print()
            print(f"train epochs: {train_loader.get_epochs()} | train iterations: {n} | train loss: {train_loss} | validation loss: {validation_loss}")
            if best_validation_loss is None:
                best_validation_loss = validation_loss
                best_validation_epoch = train_loader.get_epochs()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_epoch = train_loader.get_epochs()
                # Loss has improved, so save model.
                print("Saving the model parameters.")
                model_save = {
                    'model_state_dict': rnn.state_dict(),
                    'validation_loss': best_validation_loss,
                    'epoch': best_validation_epoch,
                }
                torch.save(model_save, config.save_model_params_file)
            print(f"best validation loss: {best_validation_loss} | occured at epoch: {best_validation_epoch}")
            x_specgram = x_specgram[0, :, :]
            max_plot_val = x_specgram.max().item()
            plot_image_matrix(x_specgram, file_name=f"{config.results_dir}/in_features.png", 
                                  title="in_features", 
                                  origin="lower",
                                  vmin=0.0,
                                  vmax=max_plot_val)

            y_specgram_plot = y_specgram[0, :, :]
            plot_image_matrix(y_specgram_plot, file_name=f"{config.results_dir}/target_features.png", 
                                  title="target_features", 
                                  origin="lower",
                                  vmin=0.0,
                                  vmax=max_plot_val)

            y_specgram_pred_plot = y_specgram_pred[0, :, :]
            plot_image_matrix(y_specgram_pred_plot, file_name=f"{config.results_dir}/predicted_out_features.png", 
                                  title="predicted_out_features", 
                                  origin="lower",
                                  vmin=0.0,
                                  vmax=max_plot_val)
            print()

            train_loss_accumulator.reset()
            
            # Plot the training and validation loss:
            if len(train_loss_list) > 1:
                plot_wrapper(train_epoch_list, train_loss_list, file_name=f"{config.results_dir}/train_loss.png", 
                             title="Training loss",
                             x_label="Epochs", 
                             y_label="Loss")
                plot_wrapper(train_epoch_list, validation_loss_list, 
                             file_name=f"{config.results_dir}/validation_loss.png", 
                             title="Validation loss",
                             x_label="Epochs", 
                             y_label="Loss")

            # Reset epoch count
            validation_loader.reset()
            rnn.train(True)
            validation_evaluator.print_stats(do_plots=True, description="validation")
        n += 1
    print("Done training.")


def evaluate_musdb18_rnn():
    """Evaluate on MUSDB18 test set.

    
    """
    config = config_musdb18_rnn
    if True:
        # prevent GPU out of memory error.
        config.batch_size = 10
    print(f"Using device: {config.device}")
    if config.rnn_type == "vanillaRNN":
        rnn = VanillaRNN(config, config.feature_dim, config.hidden_dim, config.feature_dim,
                         recurrent_drop_prob=config.recurrent_drop_prob,
                            input_drop_prob=config.input_drop_prob,
                            enable_bias = True,
                            enable_input_layernorm=False,
                            enable_state_layernorm=True)
    elif config.rnn_type == "FactorizedRNN":
        rnn = FactorizedRNN(config, config.feature_dim, config.hidden_dim, config.feature_dim)
    else:
        raise ValueError(f"Unrecognized value for config model_name: {config.rnn_type}")
    
    print("Loading the saved model parameters.")
    model_load = torch.load(config.save_model_params_file)
    rnn.load_state_dict(model_load['model_state_dict'])
    print(f"Model's best validation loss during training: {model_load['validation_loss']} occured at epoch: {model_load['epoch']}")
    rnn = rnn.to(config.device)

    print("Loading test dataset")
    test_loader = FastMusDBDatasetLoader(subsets="test", batch_size=config.batch_size,
                                          batch_sample_count=config.audio_clip_length_seconds*config.audio_sample_rate_hz,
                                          split=None,
                                          target="vocals",
                                          root=config.musdb_root)
    test_loss_accumulator = ValueAccumulator()
    test_evaluator = BasicRNNEvaluator(rnn, device=config.device)

    example_count = 0
    sdr_sum = 0
    while test_loader.get_epochs() < 1:
        (x, y) = test_loader.get_batch()
        x = np2t(x)
        y = np2t(y)
        x = x.to(config.device)
        y = y.to(config.device)
        # x an y have shape = (batch_size, num_channels, num_samples)
        with torch.no_grad():
            (x_phase, x_specgram) = samples_to_spectrograms2(x, config)
            (y_phase, y_specgram) = samples_to_spectrograms2(y, config)
        
            y_specgram_pred, loss = test_evaluator.forward(
                    x_specgram,
                    y_specgram,
                    enable_backprop_through_time=config.enable_backprop_through_time,
                    loss_type=config.loss_type,
            )
            y_waveform_predicted = spectrograms_to_samples(y_phase, y_specgram_pred, config)
            # compute the metrics
            sdr = fast_bss_eval.sdr(y, y_waveform_predicted, 
                                    load_diag=1e-5)
            if example_count == 0:
                def select_example_for_save_audio(sig, example_index):
                    sig = sig[example_index, :]
                    sig = torch.unsqueeze(sig, 0)
                    sig = sig.to('cpu')
                    return sig
                
                # Save only the first example:
                print(f"Saving example: {example_count}")

                example_index = 0
                y_pred_save = select_example_for_save_audio(y_waveform_predicted, example_index)
                y_save = select_example_for_save_audio(y, example_index)
                x_save = select_example_for_save_audio(x, example_index)

                #
                torchaudio.save("z_reconstructed_audio.wav", y_pred_save, config.sample_rate_hz)
                torchaudio.save("z_target_audio.wav", y_save, config.sample_rate_hz)
                torchaudio.save("z_input_audio.wav", x_save, config.sample_rate_hz)                
            # Get rid of -inf values before computing the mean:
            is_inf = torch.isinf(sdr)
            not_inf_mask = torch.logical_not(is_inf)
            filtered_sdr = sdr[not_inf_mask]
            sdr_sum += filtered_sdr.sum().item()
            example_count += torch.numel(filtered_sdr)
            
        if False:
            # Saving the reconstructed audio
            torchaudio.save("reconstructed_audio.wav", y_waveform_predicted, config.sample_rate)
        test_loss_accumulator.accumulate(loss.item())
        sys.stdout.write(".")
        sys.stdout.flush()
    test_loss = test_loss_accumulator.get_mean()
    print()
    print(f"Test loss: {test_loss}")
    mean_sdr = sdr_sum/example_count
    print(f"Test SDR = {mean_sdr}")
    
    print("Done.")


if __name__ == "__main__":
    # torch.set_num_threads(16)
    # torch.set_flush_denormal(True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    
    train_musdb18_rnn() # train model
    evaluate_musdb18_rnn() # evaluate model on test set

