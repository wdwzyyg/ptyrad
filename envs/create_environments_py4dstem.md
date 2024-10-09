### Method 1: Follow py4dstem documentation
https://github.com/py4dstem/py4DSTEM/edit/dev/README.md

```
conda update conda
conda create -n py4dstem python=3.10 
conda activate py4dstem
conda install -c conda-forge py4dstem pymatgen jupyterlab
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 cupy 
```
If cupy cannot be solved via conda, install from PyPl:
```
pip install cupy-cuda11x
```

### Method 2: Use pip install from the requirements_py4dstem.txt file
Then install the GPU related packages using
```
pip install -r /env/requirements_py4dstem.txt
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 cupy
```
