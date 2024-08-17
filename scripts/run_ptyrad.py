# Python script to run PtyRAD
# Created by Chia-Hao Lee on 2024.05.22

# Import
import os
import sys
import argparse

import torch

GPUID = 0
DEVICE = torch.device('cuda:' + str(GPUID))
print('Execution device: ', DEVICE)
print('PyTorch version: ', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version: ', torch.version.cuda)
print('CUDA device: ', torch.cuda.get_device_name(GPUID))
print('Current working dir: ', os.getcwd())

PATH_TO_PTYRAD = '/home/fs01/cl2696/workspace/ptyrad' # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params
from ptyrad.reconstruction import PtyRADSolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run PtyRAD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--params_path', type=str)
    parser.add_argument('--hypertune', action='store_true') # Default would be false
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    params        = load_params(args.params_path)
    ptycho_solver = PtyRADSolver(params, 
                                 if_hypertune=args.hypertune, 
                                 if_quiet=args.quiet, 
                                 device=DEVICE)
    ptycho_solver.run()