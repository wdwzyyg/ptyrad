# Installation

*PtyRAD* package is available on both `PyPI` and `conda-forge`, while the source codes are hosted on `GitHub` with additional demo params files and demo scripts.

## Platform and hardware compatibility

*PtyRAD* uses PyTorch as the computation backend, which supports a wide range of machines as shown below.

Assuming you have created and activated a fresh Python environment for *PtyRAD*, click on the corresponding {fas}`check text-success` to jump to your options:

|                  | {fab}`windows` Windows                        | {fab}`apple` macOS                           | {fab}`linux` Linux                           |
|:----------------:|:---------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| CPU              | [{fas}`check text-success`](#install-cpu)     | [{fas}`check text-success`](#install-macos)  | [{fas}`check text-success`](#install-cpu)    |
| NVIDIA GPU       | [{fas}`check text-success`](#install-nvidia)  | n/a                                          | [{fas}`check text-success`](#install-nvidia) |
| AMD GPU          |  {fas}`times text-danger`                     | n/a                                          | [{fas}`check text-success`](#install-amd)    |
| Apple GPU        | n/a                                           | [{fas}`check text-success`](#install-macos)  |   n/a                                        |

## Installation options

:::::::{tab-set}

::::::{tab-item} NVidia

**Windows and Linux system with a CUDA-supported GPU (NVidia)**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```
:::{dropdown} Note: Specify CUDA version
PyTorch ships with the specified CUDA runtime (the `/cu118` in the url) so please make sure your machine has a compatible GPU driver. Run `nvidia-smi` in your terminal to see the maximum supported CUDA runtime version. You can modify the url if your machine supports newer CUDA runtime.
:::

::::

::::{tab-item} conda

```bash
conda install ptyrad pytorch pytorch-cuda=11.8 -c nvidia -c pytorch -c conda-forge
```

:::{dropdown} Note: Specify CUDA version
PyTorch ships with the specified CUDA runtime (the `pyturch-cuda=11.8`) so please make sure your machine has a compatible GPU driver. Run `nvidia-smi` in your terminal to see the maximum supported CUDA runtime version. You can modify the `pyturch-cuda=X.Y` if your machine supports newer CUDA runtime.
:::

::::

:::::

::::::


::::::{tab-item} AMD

**Linux with an ROCm-supported GPU (AMD)**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall
```
:::{dropdown} Note: Specify ROCm version
PyTorch ships with the specified ROCm runtime (the `/rocm6.3` in the url) so please make sure your machine has a compatible GPU driver. Run `amd-smi` in your terminal to see the maximum supported ROCm runtime version. You can modify the url if your machine supports newer ROCm runtime.
:::

::::

::::{tab-item} conda

`conda-forge` community is still working on making compatible PyTorch build with ROCm (AMD) GPUs. It may or may not happen, please use `pip` for now, or follow progress [here](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/198).

::::

:::::

::::::

::::::{tab-item} CPU

**Windows / Linux without a GPU**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
```

::::

::::{tab-item} conda

```bash
conda install ptyrad -c conda-forge
```

::::

:::::

::::::

::::::{tab-item} MacOS

**MacOS with or without Apple Silicon GPU**

:::::{tab-set}

::::{tab-item} pip

```bash
pip install ptyrad
```

:::{dropdown} Note: macOS with Apple Silicon GPU
PyTorch automatically detects the macOS platform and will build with MPS (Apple Silicon) support.
:::

::::

::::{tab-item} conda

```bash
conda install ptyrad -c conda-forge
```

:::{dropdown} Note: macOS with Apple Silicon GPU
PyTorch automatically detects the macOS platform and will build with MPS (Apple Silicon) support.
:::

::::

:::::

::::::

:::::::

<!-- Invisible references just to suppress Sphinx warning, we have used JavaScript -->
(install-cpu)=
(install-macos)=
(install-nvidia)=
(install-amd)=