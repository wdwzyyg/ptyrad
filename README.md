## PtyRAD is now in internal beta test so it's limited to only the Muller group and invited people
*** Please do NOT share any part of the package to anyone else! ***

# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation

This package performs ptychographic reconstruction on 4D-STEM data using an automatic differation approach.

## Introduction

Ptychographic reconstruciton is often solved as an optimization problem using gradient descent methods. Typical ptychographic reconstruction packages (e.g. PtychoShelves, PtyPy, py4DSTEM) use an analytically derived or approximated gradients and apply them for the updates, while `ptyrad` utilizes [automatic differention](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) to automatically calculate the needed gradients. 

The main difference is that automatic differentiation allows simpler implementation for adding and modifying new optimizable variables. For typcial packages utilize analytical gradients, adding a new optimizable variable (like adding probe position correction or adaptive beam tilt) requires deriving the corresponding gradient with respect to the objective (loss) funciton. Manually deriving the gradients for new variables can be a tedious and daunting task. On the other hand, automatic differentiation provides instant gradients as long as a differentialble forward model is provided, and flexible control over the optimizable variables including optimizing the amplitude and phase of the object individually.

Additionally, automatic differentiation is the backbone of [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), which is the enabling training algorithm for all modern deep learning networks. `PtyRAD` uses [PyTorch](https://pytorch.org/)'s `autograd` for its AD architecture, which allows us to take advantages from the active PyTorch community for potentially extended capabilities and improved performance over the time. 

## Getting Started

### PtyRAD Major dependencies:

* Python 3.11 or above
* CUDA-supported GPU that supports CUDA 11 or above
* PyTorch 2.1 or above

> **Choose one of the following that matches your need. Installation via spec-file.txt is recommended.**

### 1. Install [miniconda](https://docs.anaconda.com/miniconda/) or [Anaconda](https://docs.anaconda.com/anaconda/install/) so we can create environments with `conda`

### 2a. Create Conda environment on Windows via spec-file.txt 
```bash
conda create -n ptyrad --file ./docs/spec-file_ptyrad_optuna_windows.txt
```

### 2b. Create Conda environment on Linux via spec-file.txt
```bash
conda create -n ptyrad --file ./docs/spec-file_ptyrad_optuna_linux.txt
```

Note: `ptyrad` can be changed to your preferred conda environment name, and `./docs/spec-file_xxx.txt` refers to the path to the spec-file.txt.

### 2c. Create Conda environment via specified package
```bash
conda create -n ptyrad python=3.11 matplotlib scikit-learn scipy h5py tifffile pytorch=2.1.0 torchvision optuna=3.6.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
```

*** Note ***
- PyTorch on Windows only supports Python 3.8-3.11 as of Sept. 2024.
- Creating environment with `spec-file.txt` (2a, 2b) is suggested. `conda` could be taking ~ 10-30 min to solve the environment for package versions if you go with option 2c, and could still run into package version issues 
-  The `pytorch-cuda` must match your CUDA installation, check it with `nvidia-smi` from the terminal


### 3. Installing PtyRAD

Option 1: Clone from this github repo (It's currently a private repo so may not work)

```bash
git clone https://github.com/chiahao3/ptyrad.git
```

Option 2: Download the [zip file](https://github.com/chiahao3/ptyrad/archive/refs/heads/main.zip) from this repo and unzip to your desired directory

### 4. Trying the demo in /scripts
- `run_ptyrad_quick_example.ipynb` provides the easiest interfact to run ptyrad with a specified .yml params file
- `run_ptyrad_detailed_walkthrough.ipynb` gives a gentle walkthrough from initialization, model, loss function, constraints, and to the final reconstruciton.

## Support
If you run into problems, have questions or suggested features / modifications, please create an issue [here](https://github.com/chiahao3/ptyrad/issues/new/choose).

## Learning resource
You can find previous tutorial recordings and slides from this [Box link](https://cornell.box.com/s/n5balzf88jixescp9l15ojx7di4xn1uo)

## Authors

Chia-Hao Lee (cl2696@cornell.edu)

## License
(I haven't decided which license to go with, but it's likely to be fully open just like `py4dstem`)


## Acknowledgments
This package gets inspiration from lots of community efforts, and specifically from the following packages. Some of the functions in `ptyrad` are directly translated or modified from these packages as noted in their comments to give explicit acknowledgment.
* [PtychoShelves](https://journals.iucr.org/j/issues/2020/02/00/zy5001/index.html)
* [fold_slice](https://github.com/yijiang1/fold_slice)
* [py4dstem](https://github.com/py4dstem/py4DSTEM)
* [adorym](https://github.com/mdw771/adorym)
* [SciComPty](https://www.mdpi.com/2410-3896/6/4/36)