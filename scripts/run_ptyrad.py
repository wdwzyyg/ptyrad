# Python script to run PtyRAD
# Created by Chia-Hao Lee on 2024.04.18

# Import
from random import shuffle
import torch
GPUID = 0
DEVICE = torch.device('cuda:' + str(GPUID))
print('Execution device: ', DEVICE)
print('PyTorch version: ', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version: ', torch.version.cuda)
print('CUDA device:', torch.cuda.get_device_name(GPUID))

import sys
PATH_TO_PTYRAD = 'H:/workspace/p01_code/deep_ptycho/ptyrad' # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.initialization import Initializer
from ptyrad.models import PtychoAD
from ptyrad.optimization import CombinedLoss, CombinedConstraint, ptycho_recon, loss_logger
from ptyrad.utils import select_scan_indices, make_batches, make_recon_params_dict, make_output_folder, save_results

# Load params from current directory, change this to the correct params file
from params_Si_128 import exp_params, source_params, model_params, loss_params, constraint_params, NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS, output_dir, postfix

# Init variables, model, optimizer, loss, constraint, indices, batches, etc
init          = Initializer(exp_params, source_params).init_all()
model         = PtychoAD(init.init_variables, model_params, device=DEVICE)
optimizer     = torch.optim.Adam(model.optimizer_params)
loss_fn       = CombinedLoss(loss_params, device=DEVICE)
constraint_fn = CombinedConstraint(constraint_params, device=DEVICE)
pos           = model.crop_pos.detach().cpu().numpy()
indices       = select_scan_indices(exp_params['N_scan_slow'], exp_params['N_scan_fast'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE)
batches       = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE)
recon_params  = make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS)
output_path   = make_output_folder(output_dir, indices, recon_params, model, constraint_params, postfix)

# Optimization loop
loss_iters = []
for iter in range(1, NITER+1):
    
    shuffle(batches)
    batch_losses, iter_t = ptycho_recon(batches, model, optimizer, loss_fn, constraint_fn, iter)
    loss_iters.append((iter, loss_logger(batch_losses, iter, iter_t)))
    
    ## Saving
    if SAVE_ITERS is not None and iter % SAVE_ITERS == 0:
        save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, iter, batch_losses)
