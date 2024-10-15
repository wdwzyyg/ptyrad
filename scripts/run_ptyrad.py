# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.10.07

import argparse
import sys

PATH_TO_PTYRAD = "/home/fs01/cl2696/workspace/ptyrad"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402
from ptyrad.utils import print_system_info, set_accelerator, set_gpu_device # noqa: E402

if __name__ == "__main__":
    # If you want to run with GPU 0
    # python ./scripts/run_ptyrad.py --params_path "params/demo/tBL_WSe2_reconstruct.yml" --gpuid 0
    
    # If you want to run independent processes on 2 GPUs on your workstation from Bash terminal
    # parallel --delay 5 python ./scripts/run_ptyrad.py --params_path scan4_64mrad_optune.yml --gpuid ::: 0 1
    
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid",       type=int, required=False, default=None)
    args = parser.parse_args()
    
    # Set up accelerator for multiGPU/mixed-precision setting, note that thess has no effect when we launch it with just `python <script>`
    accelerator = set_accelerator()
        
    print_system_info()
    params = load_params(args.params_path)
    device = set_gpu_device(args.gpuid)
    
    ptycho_solver = PtyRADSolver(params, device=device, acc=accelerator)

    ptycho_solver.run()

