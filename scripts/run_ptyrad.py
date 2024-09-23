# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.09.21

import argparse
import sys

import torch
import torch.distributed as dist
# dist.init_process_group(backend='gloo') # gloo backend is needed to run multiGPU on Windows

# GPUID = 0
# DEVICE = torch.device("cuda:" + str(GPUID))
# print("Execution device: ", DEVICE)
if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
    print('PyTorch version: ', torch.__version__)
    print('CUDA available: ', torch.cuda.is_available())
    print('CUDA version: ', torch.version.cuda)
    print('CUDA device count: ', torch.cuda.device_count())
    print('CUDA device: ', [torch.cuda.get_device_name(d) for d in [d for d in range(torch.cuda.device_count())]])

PATH_TO_PTYRAD = "/home/fs01/cl2696/workspace/ptyrad" #"H://workspace/ptyrad"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402

if __name__ == "__main__":
    # Example usage
    # python ./scripts/run_ptyrad.py --params_path "params/tBL_WSe2.yml"
       
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()

    params = load_params(args.params_path)
    ptycho_solver = PtyRADSolver(params)
    ptycho_solver.run()
    
    # End the process properly when in DDP mode
    if dist.is_initialized():
        dist.destroy_process_group()