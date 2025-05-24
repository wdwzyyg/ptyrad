# PtyRAD: Ptychographic Reconstruction with Automatic Differentiation

*PtyRAD* performs ptychographic reconstruction using an [automatic differention](https://en.wikipedia.org/wiki/Automatic_differentiation) (AD) framework powered by [*PyTorch*](https://pytorch.org/), which enables flexible and efficient implementation of gradient descent optimization. See our [arXiv paper](https://arxiv.org/abs/2505.07814) and the [Zenodo record](https://doi.org/10.5281/zenodo.15273176) for more information and demo datasets.

## Getting Started

### Major dependencies

* Python 3.10 or above
* PyTorch 2.0 or above
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

Based on your system and whether you have GPU support, choose the appropriate installation command below and run it in your terminal. Make sure you are inside the `ptyrad/` directory so the environment YAML files are visible.

- **3a. For Windows and Linux system with CUDA-supported GPUs**

    ```bash
    conda env create -f envs/environment_ptyrad_cuda118.yml
    ```

    >💡 **Note:** PyTorch ships with the specified CUDA runtime so please make sure your machine has a compatible GPU driver. Run `nvidia-smi` in your terminal to see the maximum supported CUDA runtime version. You can modify the specifed version of `pytorch-cuda=X.Y` in the YAML file if your machine supports newer CUDA runtime.

- **3b. For MacOS users, or Windows/Linux without a GPU**

    ```bash
    conda env create -f envs/environment_ptyrad_no_cuda.yml
    ```

    >💡 **Note:** It’s completely fine to use the GPU install options even on systems without a GPU. If your *PyTorch* installation isn’t built with CUDA support, or if *PtyRAD* can't detect a compatible GPU, it will automatically fall back to running on the CPU.

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
    pip install -e . --no-deps
    ```
    >💡 **Note:** This editable install registers *PtyRAD* as a Python package and uses the source code when imported, allowing you to import it from anywhere without worrying about your working directory.
    >
    > The `--no-deps` flag prevents `pip` from installing additional dependencies, even if they're listed in `pyproject.toml`, since all dependencies are managed by `conda` in Step 3.
    >
    > This command also creates a `ptyrad.egg-info/` folder for storing metadata. You can safely ignore or delete this folder — it won't affect the installed package.

### Step 5. Try the demo script / notebook

> 💡 **Note:** You can find the two example datasets (PSO, tBL_WSe2) in the `demo/data/` folder in our [Zenodo record](https://doi.org/10.5281/zenodo.15273176), which contains a full copy of the PtyRAD codebase (v0.1.0b4) as well.

Before running the demo, please check the following:
1. Demo datasets are downloaded and placed to the correct location under `demo/data/`
2. `(ptyrad)` environment is created and activated (if you're using command line)
3. *PtyRAD* is installed in the `(ptyrad)` environment as outlined in Step. 4
   
Now you're ready to run a quick demo using one of two interfaces:

- **5a. Interactive Jupyter interface (Recommended)**
 
    Use `run_ptyrad_quick_example.ipynb` to quickly reconstruct the demo dataset in a Jupyter notebook

- **5b. Command-line interface** (like your *Miniforge Prompt* terminal)
    ```bash
    # Here we assume working directory is set at `demo/`
    ptyrad run --params_path "params/tBL_WSe2_reconstruct.yml" --gpuid 0
    ```

### Bonus Step: How do I know if my PyTorch has the GPU support?
CUDA version, GPU support, and PyTorch build across platforms can be extremely confusing, so *PtyRAD* provides handy CLI tools to help check these information for you!

Once you activated `(ptyrad)` environment and installed *PtyRAD* via `pip insall -e .`, you'll have access to the following command:

```bash
# You can run this command anywhere from your terminal, as long as `ptyrad` is installed in the environment
ptyrad check-gpu
```

This command will print out relevant information of your CUDA information if available.

## Beyond the Demo: Using *PtyRAD* with Your Own Data

*PtyRAD* is designed to be **fully driven and specified by configuration files** — no code editing required. Each reconstruction is specified using a YAML params file, which includes the location of your data, preprocessing options, experimental parameters, and all other relevant reconstruction parameters (e.g., optimizer algorithm, constraints, loss, output files, etc).

>💡 **Note:** It's strongly recommended to read through the comments provided in the demo params YAML files because they contain comprehensive information of the available features of PtyRAD. We are diligently building our documentation website and hopefully it will be available soon!

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