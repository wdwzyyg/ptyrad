2024.08.21 Chia-Hao Lee

Initially the plan was to utilize multiple GPU slices (MIG, or multi-instance GPUs) for distributed training / optimization
However, apparently MIGs can't be used for such distributed type based on the following threds.
- https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445
- https://github.com/pytorch/pytorch/issues/130181

The apparent error message when running `accelerate env` with >1 MIG slices would be:

Traceback (most recent call last):
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc2/lib/python3.11/site-packages/torch/cuda/__init__.py", line 327, in _lazy_init
    queued_call()
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc2/lib/python3.11/site-packages/torch/cuda/__init__.py", line 195, in _check_capability
    capability = get_device_capability(d)
                 ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc2/lib/python3.11/site-packages/torch/cuda/__init__.py", line 451, in get_device_capability
    prop = get_device_properties(device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc2/lib/python3.11/site-packages/torch/cuda/__init__.py", line 469, in get_device_properties
    return _get_device_properties(device)  # type: ignore[name-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1720538437738/work/aten/src/ATen/cuda/CUDAContext.cpp":49, please report a bug to PyTorch. device=, num_gpus=