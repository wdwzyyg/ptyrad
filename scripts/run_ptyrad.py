# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.09.25

import argparse
import sys

PATH_TO_PTYRAD = "./"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402
from ptyrad.utils import set_gpu_device  # noqa: E402

if __name__ == "__main__":
    # Example usage if you want to run with GPU 0
    # python ./scripts/run_ptyrad.py --params_path "params/demo/tBL_WSe2_hypertune.yml" --gpuid 0
    
    # Example usage if you want to run independent processes on 2 GPUs from terminal
    # parallel --delay 5 python ./scripts/run_ptyrad.py --params_path scan4_64mrad_optune.yml --gpuid ::: 0 1
       
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid",       type=int, required=False, default=0)
    args = parser.parse_args()
    
    device = set_gpu_device(args.gpuid)
    params = load_params(args.params_path)

    ptycho_solver = PtyRADSolver(params, device=device)
    ptycho_solver.run()