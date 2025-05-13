# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation

*PtyRAD* performs ptychographic reconstruction using an [automatic differention](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) framework powered by [*PyTorch*](https://pytorch.org/), which enables flexible and efficient implementation of gradient descent optimization. See our [arXiv paper](https://arxiv.org/abs/2505.07814) for more information.

## Getting Started

### Major dependencies

* Python 3.10 or above
* PyTorch 2.1 or above
* While *PtyRAD* can run on CPU, GPU is strongly suggested for high-speed ptychographic reconstructions. 
    - *PtyRAD* supports both NVIDIA GPUs with CUDA and Apple Silicon (MPS)
* *PtyRAD* was tested on Windows, MacOS, and Linux

### Recommended Tool (Optional)

If you're not comfortable using the terminal, we suggest installing [*Visual Studio Code*](https://code.visualstudio.com/Download).
*VS Code* makes it easier to manage Python environments, open terminals in the correct folder, and run Jupyter notebooks — all in one interface.
After installing *VS Code*, you can:

- Open the `ptyrad/` folder in *VS Code*

- Seamlessly switch between different created Python environments

- Easily launch the notebook or run scripts inside the same window

## Step-by-Step Guide 
> 💡 **Note:** All the commands in this `README.md` are single-line commands — copy and paste them as-is, even if they wrap in your PDF viewer.

### Step 1. Download or clone the *PtyRAD* GitHub repo
You can directly download the repo as a zip file and unzip it at desired location, or run

```bash
# For Windows users, you'll need to install Git first
git clone https://github.com/chiahao3/ptyrad.git ptyrad
```

This will clone the repo into a folder named `ptyrad` under your current working directory.

### Step 2. Install a Python environment management software
We recommend [*Miniforge*](https://github.com/conda-forge/miniforge), a lightweight and open-source alternative to [*Anaconda*](https://www.anaconda.com/download). It provides the conda ecosystem, while solving environments much faster and taking up much less disk space.

Once *Miniforge* is installed:

1. Open the *"Miniforge Prompt"* on Windows, or a new terminal window on MacOS/Linux.

    > 💡 **Tip:** On Windows, use the Start menu to search for *"Miniforge Prompt"*. 
    >
    > If you prefer to use *cmd* or *PowerShell* on your Windows machine, you can optionally run the following command in your *Miniforge Prompt*:
    >    ```bash
    >   # This command will allow other terminals like *cmd* or *PowerShell* to directly access `conda`
    >    conda init
    >    ```
    >   This command only need to be executed once.
    
    > 💡 **Tip:** For macOS and Linux, `conda` should be automatically available after installing *Miniforge* when you open a new terminal. If not, try restarting your terminal, or manually activate it with:
    >    ```bash
    >   # Adjust the path if you installed Miniforge elsewhere
    >    source ~/miniforge3/bin/activate
    >    ```

2. Navigate into the `ptyrad/` folder that contains this `README.md`

    You can do this using the `cd` command, which means **changing directory**:

    ```bash
    cd path/to/ptyrad
    ```

    Replace `path/to/ptyrad` with the actual path to this folder.
    This step is required so that all the example scripts and paths work correctly.

3. You can check that you're in the right folder by running:

    ```bash
    ls  # or `dir` on Windows
    ```

    You should see files like `README.md`, and other folders like `demo/`, `scripts/`, and `src/` which keeps the main codebase.

You're now ready to set up a dedicated Python environment for *PtyRAD*.
Creating separate environments for each project is strongly recommended — it helps avoid version conflicts and keeps your setup clean and reproducible.

### Step 3. Create the Python environment required for *PtyRAD*

Based on your system and whether you have GPU support, choose the appropriate installation command below and run it in your terminal.

- **3a. For Windows and Linux system with CUDA-supported GPUs**

    - Option 1 (Recommended): Create the environment using the provided YAML file

        ```bash
        # Ensure you are inside the `ptyrad/` directory so the environment YAML file is visible
        conda env create -f environment_ptyrad.yml
        ```

    - Option 2: Manually specifying the packages for more control
        
        ```bash
        conda create -n ptyrad python pytorch pytorch-cuda accelerate torchvision optuna matplotlib scikit-learn scipy h5py tifffile jupyter -c nvidia -c pytorch -c conda-forge
        ```

- **3b. For MacOS users, or Windows/Linux without a GPU**

    ```bash
    conda create -n ptyrad python pytorch accelerate torchvision optuna matplotlib scikit-learn scipy h5py tifffile jupyter -c pytorch -c conda-forge
    ```

    >💡 **Note:** It’s completely fine to use the GPU install options even on systems without a GPU. Packages like `pytorch-cuda` will simply remain inactive and unused. If your *PyTorch* installation isn’t built with CUDA support, or if *PtyRAD* can't detect a compatible GPU, it will automatically fall back to running on the CPU.

The created Python environment will be named `(ptyrad)` and stored under `miniforge3/envs/ptyrad/`. Throughout this guide, we use `(ptyrad)` to refer to the Python environment, and `ptyrad/` to refer to the repository folder. The parentheses `(...)` help keep them distinct.

### Step 4. Install the *PtyRAD* package

Once the Python environment `(ptyrad)` is created:

1. Activate the `(ptyrad)` environment:

    In your terminal, run the following command:

    ```bash
    conda activate ptyrad
    ```

    Your prompt should now begin with `(ptyrad)` to indicate the environment has been activated.

2. Navigate to the `ptyrad/` folder if you haven't done so.
   
   You can confirm you're in the right place by listing the contents and you will need `pyproject.toml` for the installation of *PtyRAD*.
   ```bash
    ls  # or `dir` on Windows
    ```
3. Install *PtyRAD* as an editable Python package inside the `(ptyrad)` environment
   ```bash
   # Note that there is a space and period after "-e"
    pip install -e .
    ```
    >💡 **Note:** This editable install registers *PtyRAD* as a Python package, so you can import it from anywhere — without worrying about your working directory or `sys.path`.
    >
    > This command also creates a `ptyrad.egg-info/` folder for storing metadata. You can safely ignore or delete this folder — it won't affect the installed package.

### Step 5. Try the demo script / notebook

Once your `(ptyrad)` environment is activated, and you've installed *PtyRAD*, you're ready to run a quick demo using one of two interfaces:

 - Interactive interface (Recommended):
 
    Use `run_ptyrad_quick_example.ipynb` to quickly reconstruct the demo dataset in a [*Jupyter notebook*](https://docs.jupyter.org/en/latest/running.html)

 - Command-line interface:
  
    Use `run_ptyrad.py` to run *PtyRAD* directly from your *Miniforge Prompt* terminal:

    ```bash
    # Here we assume working directory is set at demo/
    python ../scripts/run_ptyrad.py --params_path "params/tBL_WSe2_reconstruct.yml" --gpuid 0
    ```

We've included two example datasets (PSO, tBL_WSe2) in the `demo/data/` folder and three preconfigured parameter files in `demo/params/` to get you started. The full datasets and all parameter files used in the manuscript will be released on Zenodo.

## Beyond the Demo: Using *PtyRAD* with Your Own Data

*PtyRAD* is designed to be **fully driven and specified by configuration files** — no code editing required. Each reconstruction is specified using a YAML params file, which includes the location of your data, preprocessing options, experimental parameters, and all other relevant reconstruction parameters (e.g., optimizer algorithm, constraints, loss, output files, etc).

To reconstruct your own dataset:

1. Prepare your data and place it in any folder of your choosing (e.g., a `data/` directory in your workspace).

2. Create or edit a YAML params file with the appropriate paths and settings for your data. You can keep this file anywhere — as long as your script or notebook knows where to find it.

3. Run *PtyRAD* using the same notebook or script provided in this repo, but pointing it to your customized params file. Note that the `output/` folder will be automatically generated under your current working directory.

## Author

Chia-Hao Lee (cl2696@cornell.edu)

Developed at the Muller Group, Cornell University.

## Acknowledgments

Besides great support from the entire Muller group, this package gets inspiration from lots of community efforts, and specifically from the following packages. Some of the functions in *PtyRAD* are directly translated or modified from these packages as noted in their docstrings/comments to give explicit acknowledgment.
* [PtychoShelves](https://journals.iucr.org/j/issues/2020/02/00/zy5001/index.html)
* [fold_slice](https://github.com/yijiang1/fold_slice)
* [py4dstem](https://github.com/py4dstem/py4DSTEM)
* [adorym](https://github.com/mdw771/adorym)
* [SciComPty](https://www.mdpi.com/2410-3896/6/4/36)