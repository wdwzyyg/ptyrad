# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2024.10.19

import argparse
import sys

PATH_TO_PTYRAD = "/home/fs01/cl2696/workspace/ptyrad"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.data_io import load_params  # noqa: E402
from ptyrad.reconstruction import PtyRADSolver  # noqa: E402
from ptyrad.utils import CustomLogger, print_system_info, set_accelerator, set_gpu_device # noqa: E402

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
    parser.add_argument("--jobid",       type=int, required=False, default=0)
    args = parser.parse_args()
    # for weight in [0.1, 0.06, 0.03, 0.01]:
    #     for batch in [1024,256,64,16]:
    #         for pmode in [6]:
    for idx in range(40):
        try:
            # Set up custom logger
            logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_date=True, prefix_jobid=args.jobid, append_to_file=True, show_timestamp=True)
            
            # Set up accelerator for multiGPU/mixed-precision setting, note that thess has no effect when we launch it with just `python <script>`
            accelerator = set_accelerator()
                
            print_system_info()
            params = load_params(args.params_path)
            device = set_gpu_device(args.gpuid)
                            
            # Run ptyrad_ptycho_solver
            print(f"Running hypertune round {idx}")
            # print(f"Running batch = {batch}, pmode = {pmode}, loss_sparse weight = {weight}")
            # params['exp_params']['pmode_max'] = pmode
            # params['recon_params']['BATCH_SIZE']['size'] = batch
            # params['loss_params']['loss_sparse']['weight'] = weight
            params['hypertune_params']['study_name'] += str(idx).zfill(2)
            params['recon_params']['postfix'] += str(idx).zfill(2)
            
            ptycho_solver = PtyRADSolver(params, device=device, acc=accelerator, logger=logger)
            

            ptycho_solver.run()
            
        except Exception as e:
            # print(f"An error occurred for batch={batch}, pmode={pmode}: {e}")
            print(f"An error occurred for hypertune round {idx}: {e}")


