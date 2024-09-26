# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.09.26

import argparse
import sys

import torch.distributed as dist

PATH_TO_PTYRAD = "./"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402
from ptyrad.utils import set_gpu_device  # noqa: E402

if __name__ == "__main__":
    # If you want to run with GPU 0
    # python ./scripts/run_ptyrad.py --params_path "params/demo/tBL_WSe2_reconstruct.yml" --gpuid 0
    
    # If you want to run independent processes on 2 GPUs on your workstation from Bash terminal
    # parallel --delay 5 python ./scripts/run_ptyrad.py --params_path scan4_64mrad_optune.yml --gpuid ::: 0 1
    
    # If you want to run a multi-GPU (2) reconstruction with mixed-precision via `accelerate`
    # accelerate launch --multi_gpu --num_processes=2 --mixed_precision='fp16' ./scripts/run_ptyrad.py --params_path "params/demo/tBL_WSe2_reconstruct.yml"
       
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid",       type=int, required=False, default=None)
    args = parser.parse_args()
    
    params = load_params(args.params_path)
    device = set_gpu_device(args.gpuid)
    
    if args.gpuid is not None:
        ptycho_solver = PtyRADSolver(params, device=device)
    else:
        # This will let `accelerate` to automatically choose the device based on `accelerate config`
        ptycho_solver = PtyRADSolver(params, device=None)

    ptycho_solver.run()
    
    # End the process properly when in DDP mode
    if dist.is_initialized():
        dist.destroy_process_group()
