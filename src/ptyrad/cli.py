## Command line Interface module

import argparse

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import (
    CustomLogger,
    print_gpu_info,
    print_system_info,
    set_accelerator,
    set_gpu_device,
)


def run(args):

    # Setup CustomLogger
    logger = CustomLogger(
        log_file='ptyrad_log.txt',
        log_dir='auto',
        prefix_date=True,
        prefix_jobid=args.jobid,
        append_to_file=True,
        show_timestamp=True
    )

    # Set up accelerator for multiGPU/mixed-precision setting, 
    # note that these we need to call the command as:
    # `accelerate launch --num_processes=2 --mixed_precision=fp16 ptyrad run <ARGUMENTS>`
    accelerator = set_accelerator() 

    print_system_info()
    params = load_params(args.params_path)
    device = set_gpu_device(args.gpuid)
    ptycho_solver = PtyRADSolver(params, device=device, acc=accelerator, logger=logger)
    ptycho_solver.run()


def check_gpu(args):
    print_gpu_info()


def print_info(args):
    print_system_info()


def validate_params(args):
    try:
        params = load_params(args.params_path)
        print("Parameters are valid.")
    except Exception as e:
        print(f"Invalid parameters: {e}")


def gui(args):
    print("[placeholder] GUI not implemented yet.")


def main():
    parser = argparse.ArgumentParser(
        description="PtyRAD Command-Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    parser_run = subparsers.add_parser("run", help="Run PtyRAD reconstruction")
    parser_run.add_argument("--params_path", type=str, required=True)
    parser_run.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser_run.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    parser_run.set_defaults(func=run)

    # check-gpu
    parser_check_gpu = subparsers.add_parser("check-gpu", help="Check GPU availability")
    parser_check_gpu.set_defaults(func=check_gpu)

    # print-system-info
    parser_info = subparsers.add_parser("print-system-info", help="Print system info")
    parser_info.set_defaults(func=print_info)

    # validate-params (placeholder) #TODO
    parser_validate = subparsers.add_parser("validate-params", help="Validate parameter file (not implemented)")
    parser_validate.add_argument("--params_path", type=str, required=True)
    parser_validate.set_defaults(func=validate_params)

    # gui (placeholder) #TODO 
    parser_gui = subparsers.add_parser("gui", help="Launch GUI (not implemented)")
    parser_gui.set_defaults(func=gui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
