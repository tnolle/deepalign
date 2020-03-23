# DeepAlign: Alignment-based Process Anomaly Correction Using Recurrent Neural Networks

This repository holds an efficient implementation of the DeepAlign algorithm as proposed in the paper.
The code in this repository can be used to reproduce the results given in the paper.

## Additional Material

To illustrate the findings in our paper, this repository contains Jupyter notebooks.

1. [Paper Process from Sec. 4](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/1.%20Paper%20Process%20from%20Sec.%204.ipynb)
    * Describes the creation of the paper process used as the running example in the paper.
2. [Dataset Generation](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/2.%20Dataset%20Generation.ipynb)
    * Downloads the pretrained models and datasets used in the evaluation. Also includes the dataset generation script.
    * [2.A1 Generation Algorithm](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/2.A1%20Generation%20Algorithm.ipynb) explains how the generation algorithm works.
3. [Training the Models](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/3.%20Training%20the%20Models.ipynb)
    * Demonstartes how to train your own models.
4. [Alignments](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/4.%20Alignments.ipynb)
    * This notebook contains all the examples from the Evaluation section of the paper and outlines how to reproduce them.
5. [Caching the Alignments](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/5.%20Caching%20the%20Alignments.ipynb)
    * This is a helper script to speed up the evaluation.
6. [Evaluation Script](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/6.%20Evaluation%20Script.ipynb)
    * This is the evaluation script used in the paper.
7. [Evaluation](https://nbviewer.jupyter.org/github/tnolle/deepalign/blob/master/notebooks/7.%20Evaluation.ipynb)
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
1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: Unsupervised anomaly detection in noisy business process event logs using
    denoising autoencoders, 2016](https://doi.org/10.1007/978-3-319-46307-0_28)
2. [Nolle, T., Luettgen, S., Seeliger A., Mühlhäuser, M.: Analyzing business process anomalies using autoencoders,
    2018](https://doi.org/10.1007/s10994-018-5702-8)
3. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning,
    2018](https://doi.org/10.1007/978-3-319-98648-7_16)
4. [Nolle, T., Luettgen, S., Seeliger, A., Mühlhäuser, M.: BINet: Multi-perspective Business Process Anomaly Classification,
   2019](https://doi.org/10.1016/j.is.2019.101458)
