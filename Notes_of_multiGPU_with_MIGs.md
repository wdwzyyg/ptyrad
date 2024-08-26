
# MIG
2024.08.21 Chia-Hao Lee

Initially the plan was to utilize multiple GPU slices (MIG, or multi-instance GPUs) for distributed training / optimization
However, apparently MIGs can't be used for such distributed type based on the following threds.
- https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445
- https://github.com/pytorch/pytorch/issues/130181

The apparent error message when running `accelerate env` with >1 MIG slices would be:

### Error message
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

# Multi-GPU
2024.08.26 Chia-Hao Lee

Although we successfully get accelerate to run with PtychoAD class, it seems that it's running 2 same reconstructions on 2 GPUs. 
The log message are all duplicated. The problem is probably the data loader or CBED at different probe positions didn't get parallelized. 
We'll probably need to refactor the optimization loop, and make data loader a separate class object so that it can get parallelized as well.
This probably make better sense because the multiGPU plan is to split the batches to multiple GPUs, so we need a centralized dataloader.
Another possible thing to try is to remove excessive layers of abstraction and to debug the issue. Quite frankly it might not worth the effort because
we don't have such GPU clusters at Cornell as of now. Running on a single GPU is mostly fine.

### Log message
Done batch 1 in 0.017 sec
Done batch 205 in 0.017 sec
Done batch 103 in 0.017 sec
Done batch 307 in 0.017 sec
Done batch 205 in 0.017 sec
Done batch 409 in 0.017 sec
Done batch 307 in 0.017 sec
Done batch 512 in 0.017 sec
Apply ortho pmode constraint at iter 14, relative pmode power = [0.884 0.047 0.035 0.034], probe int sum = 451.1710
Apply fix probe int constraint at iter 14, probe int sum = 450.4549
Apply hard positivity constraint on objp at iter 14
Iter: 14, Total Loss: 0.2520, loss_single: 0.2512, loss_poissn: 0.0000, loss_pacbed: 0.0000, loss_sparse: 0.0008, loss_simlar: 0.0000, in 0.0 min 8.735599 sec

Done batch 1 in 0.017 sec
Done batch 409 in 0.017 sec
Done batch 103 in 0.017 sec
Done batch 512 in 0.017 sec
Apply ortho pmode constraint at iter 14, relative pmode power = [0.879 0.048 0.042 0.031], probe int sum = 452.0975
Apply fix probe int constraint at iter 14, probe int sum = 450.4549
Apply hard positivity constraint on objp at iter 14
Iter: 14, Total Loss: 0.2518, loss_single: 0.2510, loss_poissn: 0.0000, loss_pacbed: 0.0000, loss_sparse: 0.0008, loss_simlar: 0.0000, in 0.0 min 8.842690 sec