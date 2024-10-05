# Notes of multi-GPU for ptychorgaphy
Iterative algorithms for ptychography is very computational heavy even with GPU. Popular packages like PtychoShelves and py4DSTEM use only 1 GPU, while multi-GPU can "potentially" provide 2 benefits:
1. More VRAM can enable larger reconstruction jobs (larger dataset, larger batch size, etc) that might not fit into a single GPU
2. Accelerate the iteration time by spreading the workload on more CUDA cores

Although multi-GPU might seem attempting, the reality is:
1. It's much harder to code, maintain, and make it compatible with different hardware setup from different users.
2. Typical iterative ptychorgaphic reconstruction algorithms require fairly small VRAM even if we load the entire dataset. 20 GB of VRAM is very sufficient unless you're doing ptychotomo reconstruction. Even you do, `gradient accumulation` (introduced in PtyRAD beta2.5) can probably reduce most of the memory requitement. Therefore, using multiple GPU for more VRAM is more relevant in ML community than in ptychorgaphy.
3. Due to the GPU-GPU inter-communication is usually much slower than the internal VRAM-CUDA inner communication in a single GPU, the multi-GPU speed up is usually sublinear, and I've been getting 1.3-1.7x for common ML models. `NVLink` is arguably a must have for multi-GPU setup and it's only available for data center cards. Better initialization, better optimizer, or just fine tune the learning rate might get you much more meaningful speedful of ptychogrphic convergence.

Due to the platform limitation, currently Windows doesn't have NCCL (NVIDIA Collective Communications Library) support so we can't do multi-GPU communication. **Therefore, multi-GPU reconstruction is not supported on Windows. This is not an accelerate or PyTorch problem, it's an NVidia problem and I have no idea whether they have any plan on it.** `PtyRAD` now has both `main` and `accelerate` branches, while `main` is meant to run without the need to install `accelerate`, the latest commit of `accelerate` branch fixes the dependency so that `accelerate` branch can now run single-GPU tasks without installing `accelerate`. In other words, I might merge the `accelerate` branch into `main` in the near future so that I only need to maintain one branch. Although Windows users would not be able to use multi-GPU, they can still run the same code on single GPU using an environment without `accelerate`. On the other hand, Linux users would be able to try out multi-GPU if they have the environment and hardware setup.

Chia-Hao Lee, cl2696@cornell.edu
Last update: 2024.10.02

---

## Current status
- 2024.09.23: Code runs for python, accelerate with 1 GPU, accelerate with 2 GPUs, but no speed up. I'm suspecting it's the `split_batches` not working on my custom `BatchesDataset`. Tested til 5AM and confirms it's the `BatchesDataset` and `split_batches` were not set correctly. With `IndicesDataset` it's running more correctly, although it's indeed slower than 1 GPU. Another issue is the reconstruction is incorrect when run via `accelerate`, turns out it's the Complex probe not handled correctly in PyTorch 2.4 even though it didn't complain. Need to use the `torch.view_as_complex` trick to get away with it.
- 2024.09.24: Clean up the code. It's running smoothly locally in jupyter notebook, in cluster via python or accelerate. Model saving/loading without any issue. Add the `mixed_precision_type` support. Tried the `base_precision_type` for fix precision attempt for 'bf16' and 'fp16' but it would get stuck at the backward part. No solution yet, fall back to `amp` seems to be the only option.
- 2024.09.25: Implemented the accelerate enabled mixed precision and make it a CLI argument for simplicity. The original grad accumulation implementation seems reasonable with split_batches=True.
- 2024.09.26: Experimentally merge the `accelerate` into `dev`. Clean up the code structure and driver script logic. Fix the error in `make_save_dict` because the saved probe is in the real view (pmode, Ny, Nx, 2) instead of the complex view.
- 2024.09.30: Decided to keep the multi-GPU inside the `accelerate` branch for now because I'm a bit concerned about adding a new dependency would break the platform compatibility. Also `accelerate` package at conda-forge channel (0.21) is a bit too old for my current implementation. Besides, I'm not sure if it's possible to do multiple GPU on Windows through this setup so might try to keep `main` branch to be platform independent.
- 2024.10.01: Managed to get the `accelerate` branch running on Windows environment without `accelerate` package installed. This partially solves the dependency issue, but we still can't run `accelerate`'s multi-GPU feature on Windows even we install the `accelerate` package on Windows and set the backend to 'gloo'. 'gloo' is really designed for distributed CPU training so there's no point installing accelerate on Windows if you're planning to do multi-GPU. It's just unfortunately not supported. See [here](https://pytorch.org/docs/stable/distributed.html).
- 2024.10.05: Feel like it's about time to merge the `accelerate` branch into `main` after quite some testing. If anything, an environment without `accelerate` can still work fine.

## multi-GPU speed up table
- I did quick tests using the full A100 node with tBL-WSe2 dataset
- 128x128x128x128 4D-dataset, 6 probe modes with 12 slices of 1 Ang, 1 object mode
- Batch size from 32 to 4096, no gradient accumulation
- 1 shot of 10 iterations, save figures every 5 iters (saving is slow so I'm only reporting the averaged iter time)

![Iteration_time_vs_batch_size](./docs/20240924_multi-GPU/iteration_time_vs_batch_size.png)

## Some multi-GPU todo
- Update the environment requirement file
- Find workaround to actually use the `GROUP_MODE` indices for multi-GPU (currently the DataLoader can only do random indices)

## Lesson learned
- PyTorch hasn't support distributed training (multi GPU) on MIG yet. See [here](https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445) and [here](https://github.com/pytorch/pytorch/issues/130181)
- `accelerate` from HuggingFace is essentially a wrapper over PyTorch's DDP (DistributedDataParallel), and DDP has quite some details.
- Note that when `accelerate` fails to run things in DDP it "may not" throw any error......
- `amp` doesn't support ComplexFloat yet, but seems like `torch.view_as_real` can work!
- Although most operation in `fp16` and `bf16` can work, the final `.backward` hasn't been successful and `amp` is probably always needed for the mixed-precision purpose
- NCCL doesn't support Windows or Mac. See [here](https://discuss.pytorch.org/t/nccl-for-windows/203543). So the workaround is to use `gloo` instead of `NCCL` as the backend as described [here] (https://discuss.pytorch.org/t/how-to-set-backend-to-gloo-on-windows/161448/3) on Windows. Do `import torch.distributed as dist` and then `dist.init_process_group(backend='gloo')` on Windows or Mac, although it's probably much easier to do things on Linux machines.
- DDP is also a wrapper over original `model`, so we can't do `model.<attribute>` directly. The error looks like `AttributeError: 'DistributedDataParallel' object has no attribute <attribute>`. Although `model.module.<attribute>` can be used to access the entry, it's not single/multi-GPU compatible anymore, a better way to handle this is to create some `get_attribute` method inside `model` and check if there's a `module` attached to handle it internally.
- PyTorch 2.1 doesn't support complex valued network for DDP, it's a NCCL issue and will give `RuntimeError: Input tensor data type is not supported for NCCL process group: ComplexFloat`. [PyTorch 2.4 handles it internally to avoid the error.](https://github.com/pytorch/pytorch/issues/71613)
- Follow up that the internal solution for Complex in NCCL did not really handle PtyRAD model correctly and gives incorrect reconstruction without throwing any error, so I have to do some `torch.view_as_complex` trick to work around
- The key to DDP or distributed training is through "device placement", `accelerate` provides `.prepare()` wrapper function to do it, but for custom PyTorch model (`torch.nn.Module`) the `.to()` method might not work as expected, because custom model is just an artificial construct that holds parameters. In order to get `.prepare(model)` or `model.to(device)` working properly, you must "register" the model parameters with either `nn.Parameter()` or `register_buffer()`. See [here] (https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723) and [here] (https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/10) for more information.
- Once registered as parameter of buffer, the tensor will be sync between processes and the datatype must be at least "single" for NCCL
- iteration-wise constraints might change the grad layout and gives you warning as `UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.` Use `params=params.contiguous()` after `.permute()` or `.reshape` to avoid the warning.
- Overall speaking, in order to get all the juice from ML community we have to code in their flavor and it's been quite a learning process. 
- `torch.compile` for JIT does not support Windows, and it also does not support complex value in the graph

### References for multi gpu ptychorgaphy
- https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-26-32082&id=306954
- https://www.nature.com/articles/s41598-022-09430-3
- https://dl.acm.org/doi/10.1145/3447818.3460380
- https://github.com/xiaodong-yu/PtyGer
- https://developer.nvidia.com/blog/accelerating-ptychography-workflows-with-nvidia-holoscan-at-diamond-light-source/

### 2024.08.21 Error message for `accelerate env` with >1 MIG slices would be:
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

### 2024.09.20 Error message of mixed precision (amp) not implemented for Complex
Traceback (most recent call last):
  File "H:\workspace\ptyrad\scripts\run_ptyrad.py", line 44, in <module>
    ptycho_solver.run()
  File "H:\/workspace/ptyrad\ptyrad\reconstruction.py", line 131, in run
    self.reconstruct()
  File "H:\/workspace/ptyrad\ptyrad\reconstruction.py", line 99, in reconstruct
    recon_loop(model, self.init, params, optimizer, self.loss_fn, self.constraint_fn, indices, batches_dl, output_path, acc=self.accelerator)
  File "H:\/workspace/ptyrad\ptyrad\reconstruction.py", line 244, in recon_loop
    batch_losses, iter_t = recon_step(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=model.verbose, acc=acc)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "H:\/workspace/ptyrad\ptyrad\reconstruction.py", line 299, in recon_step
    optimizer.step() # batch update
    ^^^^^^^^^^^^^^^^
  File "C:\Users\chiahao3\anaconda3\envs\debluro\Lib\site-packages\accelerate\optimizer.py", line 159, in step
    self.scaler.step(self.optimizer, closure)
  File "C:\Users\chiahao3\anaconda3\envs\debluro\Lib\site-packages\torch\cuda\amp\grad_scaler.py", line 410, in step
    self.unscale_(optimizer)
  File "C:\Users\chiahao3\anaconda3\envs\debluro\Lib\site-packages\torch\cuda\amp\grad_scaler.py", line 307, in unscale_
    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
                                              ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\chiahao3\anaconda3\envs\debluro\Lib\site-packages\torch\cuda\amp\grad_scaler.py", line 248, in _unscale_grads_
    torch._amp_foreach_non_finite_check_and_unscale_(
RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'ComplexFloat'

### 2024.09.21 Error message for unused parameters
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by making sure all `forward` function outputs participate in calculating loss. If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable). Parameter indices which did not receive grad for rank 1: 6. In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error

### 2024.09.24 Error message for half precision during .backward()
Traceback (most recent call last):
  File "/home/fs01/cl2696/workspace/ptyrad/./scripts/run_ptyrad.py", line 38, in <module>
    ptycho_solver.run()
  File "/home/fs01/cl2696/workspace/ptyrad/ptyrad/reconstruction.py", line 178, in run
    self.reconstruct()
  File "/home/fs01/cl2696/workspace/ptyrad/ptyrad/reconstruction.py", line 124, in reconstruct
    recon_loop(model, self.init, params, optimizer, self.loss_fn, self.constraint_fn, indices, self.dl, output_path, acc=self.accelerator)
  File "/home/fs01/cl2696/workspace/ptyrad/ptyrad/reconstruction.py", line 312, in recon_loop
    batch_losses, iter_t = recon_step(batches, grad_accumulation, model, optimizer, loss_fn, constraint_fn, niter, verbose=verbose, acc=acc)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/fs01/cl2696/workspace/ptyrad/ptyrad/reconstruction.py", line 386, in recon_step
    acc.backward(loss_batch)
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc/lib/python3.12/site-packages/accelerate/accelerator.py", line 2196, in backward
    loss.backward(**kwargs)
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc/lib/python3.12/site-packages/torch/_tensor.py", line 521, in backward
    torch.autograd.backward(
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc/lib/python3.12/site-packages/torch/autograd/__init__.py", line 289, in backward
    _engine_run_backward(
  File "/home/fs01/cl2696/anaconda3/envs/ptyrad_acc/lib/python3.12/site-packages/torch/autograd/graph.py", line 769, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Found dtype Half but expected Float
Wed Sep 25 00:42:48 EDT 2024