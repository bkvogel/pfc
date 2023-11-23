# pfc

This is the official implementation of the arxiv paper "[An NMF-Based Building Block for Interpretable Neural Networks With Continual Learning](https://arxiv.org/abs/2311.11485)".


### Setup

Requirements:
- Nvidia GPU (tested on RTX 4090, but anything recent with at least 8GB might be sufficient)
- Compatible Linux distribution (tested on Ubuntu 22.04 and Kubuntu 23.10)

Create a folder where the downloaded MNIST, Fashion MNIST, and CIFAR10 datasets will be stored. Set `datasets_root` in `run_experiments.py` to the location of this folder.

If using Anaconda, create a new conda environment and install the dependencies:

```
conda create --name pytorch_3_11 python=3.11
conda activate pytorch_3_11
```

Go to [https://pytorch.org/](https://pytorch.org/) and install Pytorch using the conda instructions. E.g., it should be similar to the following (If different, run the command from their website, not the following):

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Now install the other dependencies:

```
conda install -c conda-forge einops
conda install matplotlib
conda install -c anaconda scipy
conda install numba
```

### Organization

Main scripts:
- `run_experiments.py`: Contains all experiments except source separation.
    - To choose an experiment to run, edit the value of the `run_experiment` variable near the bottom of the file.
- `train_mss.py` Runs the source separation experiment (See below on additional setup and requirements before running)

Folders:
- `models/`: Contains the models used (PFC blocks, MLP, vanilla and factorized RNNs, etc.)
- `utils/`: Contains various utility functions, including the NMF (SGD and Lee-Seung update rules), FISTA, normalization, etc.
- `figures/`: Some experiments will output `png` files to the sub-folders in this directory, corresponding to the paper figures.
- `debug_plots/`: Some experiments will output debugging plots (as `png` files) here.
    - Any existing `png` files in this folder are automatically deleted by `run_experiments.py` just before starting the experiment.
    - These `png` plots typically refresh occasionally while the script is running. They are produced in the `print_stats()` method of the various model classes.
- `openunmix/`: Dataset loaders for the source separation experiment.
- `saved_models/`: Contains saved Pytorch model files generated while running the experiments.

Output logs:
- `experiment_results.log`: Contains concise experiment results produces by the scripts.
- `debug.log`: More verbose logs.

### Running the main experiments in `run_experiments.py`

To run these experiments:
```
python run_experiments.py
```

To view the concise output while the script it running, in another terminal:
```
tail -f experiment_results.log
```

To view more verbose output while the script it running, in another terminal:
```
tail -f debug.log
```

* Image classification experiments involving MLP, 1-block PFC, and residual 2-block PFC models: `run_experiment = 'train_and_evaluate_various_classifier'`
    - You can go into the function `train_and_evaluate_various_classifier()` and enable/disable individual experiments. E.g., set `run_mlp_image_classification_experiment = False` etc.
* Factorized RNN using standard NMF updates on a deterministic sequence memorization task: `run_experiment = 'train_and_evaluate_learning_repeated_sequence'`
* Factorized RNN using standard NMF learning and inference updates on the Copy Task: `run_experiment = 'train_and_evaluate_copy_task_factorized_rnn'`
* Conventional vanilla RNN trained with and without BPTT on the copy task: `run_experiment = 'train_and_evaluate_copy_task_vanilla_rnn'`
* Factorized and conventional vanilla RNNs on the Sequential MNIST task (uses backprop, with or without BPTT): `run_experiment = 'run_sequential_mnist_rnn_experiments'`
* Factorized RNNs on the Sequential MNIST task using only NMF learning and inference updates (no backprop): `run_experiment = 'sequential_mnist_factorized_rnn_conventional_nmf'`


### Source separation on  MUSDB18

Since the source separation experiment requires some additional dependencies, it is separated into another script: `train_mss.py`. To run it, you will need to install the following extra dependencies and download the dataset.

Install additional dependencies:

```
pip install musdb
pip install fast-bss-eval
```

Download the MUSDB18 dataset (high quality WAV version: musdb18hq).

Locate `config_musdb18_rnn` in the file `train_mss.py` and edit the following fields:
- Set `rnn_type` to either `vanillaRNN` to use the vanilla RNN or to `FactorizedRNN` to use the factorized RNN.
- Set `musdb_root` to the root folder of the downloaded MUSDB dataset.

To train and evaluate the model:

```
python train_mss.py
```

The script will print the MSE on the test dataset at the end.

### Citation

```
@misc{vogel2023nmfbased,
      title={An NMF-Based Building Block for Interpretable Neural Networks With Continual Learning}, 
      author={Brian K. Vogel},
      year={2023},
      eprint={2311.11485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```