## PtyRAD is now in internal beta test so it's limited to only the Muller group and invited people
*** Please do NOT share any part of the package to anyone else! ***

# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation

This package performs ptychographic reconstruction on 4D-STEM data using an automatic differation approach.

## Introduction

Ptychographic reconstruciton is often solved as an optimization problem using gradient descent methods. Typical ptychographic reconstruction packages (e.g. PtychoShelves, PtyPy, py4DSTEM) use an analytically derived or approximated gradients and apply them for the updates, while `ptyrad` utilizes [automatic differention](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) to automatically calculate the needed gradients. 

The main difference is that automatic differentiation allows simpler implementation for adding and modifying new optimizable variables. For typcial packages utilize analytical gradients, adding a new optimizable variable (like adding probe position correction or adaptive beam tilt) requires deriving the corresponding gradient with respect to the objective (loss) funciton. Manually deriving the gradients for new variables can be a tedious and daunting task. On the other hand, automatic differentiation provides instant gradients as long as a differentialble forward model is provided, and flexible control over the optimizable variables including optimizing the amplitude and phase of the object individually.

Additionally, automatic differentiation is the backbone of [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), which is the enabling training algorithm for all modern deep learning networks. `PtyRAD` uses [PyTorch](https://pytorch.org/)'s `autograd` for its AD architecture, which allows us to take advantages from the active PyTorch community for potentially extended capabilities and improved performance over the time. 

## Getting Started

### Dependencies

* Python 3.11 or above
* CUDA-supported GPU
* PyTorch 2.1 or above


### Create Conda envoronment
```bash
conda create -n ptyrad python=3.11 ipykernel matplotlib scikit-image scikit-learn scipy h5py tifffile pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Installing

Option 1: Clone from this github repo

```bash
git clone https://github.com/chiahao3/ptyrad.git
```

Option 2: Download the [zip file](https://github.com/chiahao3/ptyrad/archive/refs/heads/main.zip) from this repo and unzip to your desired directory

### Trying the demo
- `run_ptyrad.ipynb` gives a gentle walkthrough from initialization, model, loss function, constraints, and to the final reconstruciton.

## Support
If you run into problems, have questions or suggested features / modifications, please create an issue [here](https://github.com/chiahao3/ptyrad/issues/new/choose).

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