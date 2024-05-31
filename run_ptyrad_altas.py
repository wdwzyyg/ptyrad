# Python script to run PtyRAD
# Created by Chia-Hao Lee on 2024.05.22

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
PATH_TO_PTYRAD = 'home/fs01/cl2696/workspace/p01_code/deep_ptycho/ptyrad' # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.initialization import Initializer
from ptyrad.models import PtychoAD
from ptyrad.optimization import CombinedLoss, CombinedConstraint, ptycho_recon, loss_logger
from ptyrad.visualization import plot_summary, plot_pos_grouping
from ptyrad.utils import select_scan_indices, make_batches, make_recon_params_dict, make_output_folder, save_results, get_blob_size

# Load params from current directory, change this to the correct params file
from ptyrad.inputs.full_params_tBL_WSe2 import exp_params, source_params, model_params, loss_params, constraint_params, NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS, output_dir, prefix, postfix, fig_list

# Init variables, model, optimizer, loss, constraint
init          = Initializer(exp_params, source_params).init_all()
model         = PtychoAD(init.init_variables, model_params, device=DEVICE)
optimizer     = torch.optim.Adam(model.optimizer_params)
loss_fn       = CombinedLoss(loss_params, device=DEVICE)
constraint_fn = CombinedConstraint(constraint_params, device=DEVICE)

# Generate the indices, batches, recon_params, output_path, etc
pos          = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
probe_int    = model.opt_probe[0].abs().pow(2).detach().cpu().numpy()
dx           = exp_params['dx_spec']
d_out        = get_blob_size(dx, probe_int, output='d90') # d_out unit is in Ang
indices      = select_scan_indices(exp_params['N_scan_slow'], exp_params['N_scan_fast'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE)
batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE)
recon_params = make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS)
output_path  = make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, loss_params, prefix, postfix)

fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
fig_grouping.savefig(output_path + f"/summary_pos_grouping.png")

# Optimization loop
loss_iters = []
for niter in range(1,NITER+1):
    
    shuffle(batches)
    batch_losses, iter_t = ptycho_recon(batches, model, optimizer, loss_fn, constraint_fn, niter)
    loss_iters.append((niter, loss_logger(batch_losses, niter, iter_t)))
    
    ## Saving intermediate results
    if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
        save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses)
        
        ## Saving summary
        plot_summary(output_path, loss_iters, niter, indices, init.init_variables, model, fig_list, show_fig=False, save_fig=True)