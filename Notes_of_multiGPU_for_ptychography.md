# Notes of multiGPU for ptychorgaphy
Iterative algorithms for ptychography is very computational heavy even with GPU. Most packages utilize 1 GPU like PtychoShelves, py4DSTEM, and PtyRAD, while multi-GPU can "potentially" provide 2 benefits:
1. More VRAM can enable larger reconstruction jobs (larger dataset, larger batch size, etc) that might not fit into a single GPU
2. Accelerate the iteration time by spreading the workload on more CUDA cores

Although multiGPU might seem attempting, the reality is:
1. it's much harder to code, maintain, or make it compatible with different hardware setup from different users. 
2. Typical iterative ptychorgaphic reconstruction algorithms require fairly small VRAM even if we load the entire dataset. 20 GB of VRAM is very sufficient unless you're doing ptychotomo reconstruction. Even you do, `gradient accumulation` can probably reduce a lot of the memory usage so using multiple GPU for more VRAM is more relevant in ML community than in ptychorgaphy.
3. Due to the GPU-GPU internection is usually much slower even with NVlink, the multiGPU speed up is usually sublinear, and I've been getting 1.3-1.7x for common ML models. It might be a better use of time to try a better initialization, better optimizer, or just fine tune the learning rate if you want faster ptychogrphic convergence.

Chia-Hao Lee, cl2696@cornell.edu
Last update: 2024.09.22

---

## Current status
- 2024.09.23: Code runs for python, accelerate with 1 GPU, accelerate with 2 GPUs, but no speed up. I'm suspecting it's the `split_batches` not working on my custom `BatchesDataset`

## Lesson learned
- PyTorch hasn't support distributed training (multi GPU) on MIG yet. See [here](https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445) and [here](https://github.com/pytorch/pytorch/issues/130181)
- `accelerate` from HuggingFace is essentially a wrapper over PyTorch's DDP, and DDP has a lot of restrictions.
- Note that when `accelerate` fails to run things in DDP it "may not" throw any error......
- `amp` doesn't support ComplexFloat yet
- NCCL doesn't support Windows or Mac. See [here](https://discuss.pytorch.org/t/nccl-for-windows/203543). So the workaround is to use `gloo` instead of `NCCL` as the backend as described [here] (https://discuss.pytorch.org/t/how-to-set-backend-to-gloo-on-windows/161448/3)
- DDP is also a wrapper over original `model`, so we can't do `model.<attribute>` directly. The error looks like `AttributeError: 'DistributedDataParallel' object has no attribute <attribute>`. Although `model.module.<attribute>` can be used to access the entry, it's not single/multiGPU compatible anymore, a better way to handle this is to create a `get_attribute` method inside `model` and check if there's a `module` attached.
- PyTorch 2.1 doesn't support complex valued network for DDP, it's a NCCL issue and will give `RuntimeError: Input tensor data type is not supported for NCCL process group: ComplexFloat`. [PyTorch 2.4 handles it internally to avoid the error.](https://github.com/pytorch/pytorch/issues/71613) 
- Windows does not support `NCCL` backend and we need to do `import torch.distributed as dist` and then `dist.init_process_group(backend='gloo')`.
- The key to DDP or distributed training is through "device placement", `accelerate` provides `.prepare()` wrapper function to do it, but for custom PyTorch model (`torch.nn.Module`) the `.to()` method might not work as expected, because model is just an artificial construct that holds parameters. In order to get `.prepare(model)` or `model.to(device)` working properly, you must "register" the model parameters with either `nn.Parameter()` or `register_buffer()`. See [here] (https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723) and [here] (https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/10) for more information.
- iteration-wise constraints might change the grad layout and gives you warning as `UserWarning: Grad strides do not match bucket view strides. This may indicate grad was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.` Use `params=params.contiguous()` after `.permute()` or `.reshape` to avoid the warning.


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




