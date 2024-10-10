### CHL version on Altas:

Download the py4DSTEM repo from `benchmark` branch and cd into the working directory
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

Download the py4DSTEM repo from `benchmark` branch and cd into the working directory
```
conda create -n py4dstem python=3.11
conda activate py4dstem
conda install -c conda-forge cupy
(Optional) pip install pymatgen 
pip install -e .
```
Note: Installing cupy first would get the cudnn and cudatoolkit automatically