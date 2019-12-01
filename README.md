# DeepAlign: Alignment-based Process Anomaly Correction Using Recurrent Neural Networks

This repository holds an efficient implementation of the DeepAlign algorithm as proposed in the paper.
The code in this repository can be used to reproduce the results given in the paper.

## Additional Material

To illustrate the findings in our paper, this repository contains Jupyter notebooks.

1. [Paper Process from Sec. 4](notebooks/1.%20Paper%20Process%20from%20Sec.%204.ipynb)
    * Describes the creation of the paper process used as the running example in the paper.
2. [Dataset Generation](notebooks/2.%20Dataset%20Generation.ipynb)
    * Downloads the pretrained models and datasets used in the evaluation. Also includes the dataset generation script.
    * [2.A1 Generation Algorithm](notebooks/2.A1%20Generation%20Algorithm.ipynb) explains how the generation algorithm works.
3. [Training the Models](notebooks/3.%20Training%20the%20Models.ipynb)
    * Demonstartes how to train your own models.
4. [Alignments](notebooks/4.%20Alignments.ipynb)
    * This notebook contains all the examples from the Evaluation section of the paper and outlines how to reproduce them.
5. [Caching the Alignments](notebooks/5.%20Caching%20the%20Alignments.ipynb)
    * This is a helper script to speed up the evaluation.
6. [Evaluation Script](notebooks/6.%20Evaluation%20Script.ipynb)
    * This is the evaluation script used in the paper.
7. [Evaluation](notebooks/7.%20Evaluation.ipynb)
    * This notebook contains all tables used in the paper. It also contains some figures that didn't make it into the paper.

## Setup
The easiest way to setup an environment is to use Miniconda.

1. Install [Miniconda](https://conda.io/miniconda.html) (make sure to use a Python 3 version)
2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)
3. We suggest that you set up a dedicated environment for this project by running `conda env create -f environment.yml`
    * This will setup a virtual conda environment with all necessary dependencies.
    * If your device does have a GPU replace `tensorflow` with `tensorflow-gpu` in the `environement.yml`
4. Depending on your operating system you can activate the virtual environment with `conda activate binet` 
on Linux and macOS, and `activate deepalign` on Windows (`cmd` only).
5. If you want to make use of a GPU, you must install the CUDA Toolkit. To install the CUDA Toolkit on your computer refer to the [TensorFlow installation guide](https://www.tensorflow.org/install/install_windows).
6. If you want to quickly install the `deepalign` package, run `pip install -e .` inside the root directory.
7. Now you can start the notebook server by `jupyter notebook notebooks`.

Note: To use the graph plotting methods, you will have to install Graphviz.

## Jupyter Notebooks
Check the `notebooks` directory for example Jupyter Notebooks.
    
## References
