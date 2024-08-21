# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.08.18

import argparse
import sys

import torch

# GPUID = 0
# DEVICE = torch.device("cuda:" + str(GPUID))
# print("Execution device: ", DEVICE)
print('PyTorch version: ', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version: ', torch.version.cuda)
print('CUDA device count: ', torch.cuda.device_count())
print('CUDA device: ', [torch.cuda.get_device_name(d) for d in [d for d in range(torch.cuda.device_count())]])

PATH_TO_PTYRAD = "/home/fs01/cl2696/workspace/ptyrad"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402

if __name__ == "__main__":
    # Example usage for typical ptycho reconstruction
    # python ./scripts/run_ptyrad.py --params_path "ptyrad/inputs/full_params_tBL_WSe2.yml"

    # Example usage for hyperparameter tuning
    # python ./scripts/run_ptyrad.py --params_path "ptyrad/inputs/full_params_tBL_WSe2.yml" --hypertune
       
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--hypertune", action="store_true")  # Default would be false
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    params = load_params(args.params_path)
    ptycho_solver = PtyRADSolver(
        params, if_hypertune=args.hypertune, if_quiet=args.quiet
    )
    ptycho_solver.run()