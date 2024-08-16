# Python script to run PtyRAD
# Created by Chia-Hao Lee on 2024.05.22

# Import
from random import shuffle
import torch
import optuna

GPUID = 0
DEVICE = torch.device('cuda:' + str(GPUID))
print('Execution device: ', DEVICE)
print('PyTorch version: ', torch.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version: ', torch.version.cuda)
print('CUDA device:', torch.cuda.get_device_name(GPUID))

import sys
PATH_TO_PTYRAD = '/home/fs01/cl2696/workspace/ptyrad' # Change this for the ptyrad package path
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
constraint_fn = CombinedConstraint(constraint_params, device=DEVICE, verbose=False)
loss_fn = CombinedLoss(loss_params, device=DEVICE)

# Insert the Optuna logic

# 1. Define an objective function to be maximized.

def optuna_objective(trial):
    # 2. Suggest values of the hyperparameters using a trial object.
    
    # Turn off init printing
    init.verbose=False
    
    # Z_distance
    z_distance = trial.suggest_float('z_distance', 4, 8, step=0.5)
    # z_distance = trial.suggest_float('z_distance',z_range[0], z_range[1])
    init.init_params['exp_params']['z_distance'] = z_distance
    init.init_H()
    
    # scan_affine
    scale = trial.suggest_float('scale', 0.8, 1.2, step=0.05)
    asymmetry = trial.suggest_float('asymmetry', -0.2, 0.2, step=0.05)
    rotation = trial.suggest_float('rotation', -2,2, step=0.5)
    shear = trial.suggest_float('shear', -10, 10, step=1)
    scan_affine = (scale, asymmetry, rotation, shear)
    init.init_params['exp_params']['scan_affine'] = scan_affine
    init.init_pos()
    init.init_obj() # Update obj initialization because the scan range has changed
    
    # tilt (This will override the current tilts and force it to be a global tilt (2,1))
    tilt_y = trial.suggest_float('tilt_y', -10, 10, step=1)
    tilt_x = trial.suggest_float('tilt_x', -10, 10, step=1)
    obj_tilts = [[tilt_y, tilt_x]]
    init.init_variables['obj_tilts'] = obj_tilts

    # Create the model    
    model = PtychoAD(init.init_variables, model_params, device=DEVICE, verbose=False)

    # Use model.set_optimizer(new_lr_params) to update the variable flag and optimizer_params
    optimizer = torch.optim.Adam(model.optimizer_params)
    
    pos          = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    probe_int    = model.opt_probe[0].abs().pow(2).detach().cpu().numpy()
    dx           = init.init_variables['dx']
    d_out        = get_blob_size(dx, probe_int, output='d90', verbose=False) # d_out unit is in Ang
    indices      = select_scan_indices(init.init_variables['N_scan_slow'], init.init_variables['N_scan_fast'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE, verbose=False)
    batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE, verbose=False)
    recon_params = make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS)
    output_path  = make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, loss_params, prefix, postfix, show_init=True, verbose=False)

    fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
    fig_grouping.savefig(output_path + f"/summary_pos_grouping.png")
    
    # Actual iteration loop
    loss_iters = []
    for niter in range(1, NITER+1):
        
        shuffle(batches)
        batch_losses, iter_t = ptycho_recon(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=False) 
        loss_iter = loss_logger(batch_losses, niter, iter_t, verbose=False)
        loss_iters.append((niter, loss_iter))
        
        ## Saving intermediate results
        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            # Note that `exp_params` stores the initial exp_params, while `model` contains the actual params that could be updated if either meas_crop or meas_resample is not None
            save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses)
            
            ## Saving summary
            plot_summary(output_path, loss_iters, niter, indices, init.init_variables, model, fig_list, show_fig=False, save_fig=True, verbose=False)
            
        
        ## Pruning logic for optuna
        trial.report(loss_iter, niter)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return loss_iter

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.HyperbandPruner(),
                            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                            study_name="ptyrad_tBL_WSe2",
                            load_if_exists=True)
study.optimize(optuna_objective, n_trials=20)
print("Best params:")
for key, value in study.best_params.items():
    print(f"\t{key}: {value}")