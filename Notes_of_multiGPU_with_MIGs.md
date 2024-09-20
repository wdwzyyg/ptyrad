
# MIG
2024.08.21 Chia-Hao Lee

Initially the plan was to utilize multiple GPU slices (MIG, or multi-instance GPUs) for distributed training / optimization
However, apparently MIGs can't be used for such distributed type based on the following threds.
- https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445
- https://github.com/pytorch/pytorch/issues/130181

The apparent error message when running `accelerate env` with >1 MIG slices would be:

### Error message for MIG
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

# Revisiting multi-GPU on Windows
2024.09.20 Chia-Hao Lee

Since I solve most of the out-of-memory problem by accumulating gradients, now I want to revisit the multi-GPU possibilities.
As previously found that PtyRAD would need some refactoring to get it running with accelerate. Unfortunaly we don't have that many full GPUs on Altas.
I tried a bit harder for accelerate on Windows, although we can install it, running accelerate with multiple GPUs would need NCCL, and apparently NCCL doesn't support Windows or Mac. https://discuss.pytorch.org/t/nccl-for-windows/203543. As for now, having a Linux is probably a necessary thing.
Another workaround is to use gloo instead of NCCL as the backend. https://discuss.pytorch.org/t/how-to-set-backend-to-gloo-on-windows/161448/3
`set PL_TORCH_DISTRIBUTED_BACKEND=gloo && accelerate launch --multi_gpu --num_processes=2 ./scripts/run_ptyrad.py --params_path "ptyrad/inputs/full_params_tBL_WSe2.yml"` seems to run a little bit deeper but accelerate still wants nccl. By adding the `import torch.distributed as dist` and `dist.init_process_group(backend='gloo')` into the script, it ran all the way into reconstruction and throw an error about the amp (automatic mixed precision) so I tried reconfigure without mixed-precision. After reconfiguration it throw an error of permission denied while saving figures, it's probably my antivirus Norton or the filename being too long. Overall speaking it can run but I need to figure out how to avoid actually creating 2 instances and optimizing on 2 instances.

### Error message of mixed precision
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
