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

Note:
- Method 2 doesn't really work for me on Altas
- We should always specify the environment creation before talking about installation
- cupy seems to have some issue with python 3.10
- We shold also avoid installing from conda after pip

### CHL version on Altas:
```
conda update conda
conda create -n py4dstem python=3.11
conda activate py4dstem
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 pymatgen
pip install cupy-cuda11x 
pip install -e .
```
Note: cupy would need to be installed by pip given the selected python, cudatoolkit, and cudnn version


### CHL version on Windows:

Download the py4DSTEM repo and cd into the working directory
```
conda create -n py4dstem python=3.11
conda activate py4dstem
conda install -c conda-forge cupy
(Optional) pip install pymatgen 
pip install -e .
```

Note: Installing cupy first would get the cudnn and cudatoolkit automatically