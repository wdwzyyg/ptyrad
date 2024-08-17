## Define the ptycho reconstruction related functions

import copy
from random import shuffle
import numpy as np
import torch

from .utils import time_sync, vprint, save_results
from .visualization import plot_summary, plot_pos_grouping
from .utils import select_scan_indices, make_batches, make_output_folder, get_blob_size
from .initialization import Initializer
from .models import PtychoAD
from .optimization import CombinedLoss, CombinedConstraint

class PtyRADSolver:
    def __init__(self, params, *, if_hypertune=False, if_quiet=False, device='cuda:0'):
        self.params       = params
        self.if_hypertune = if_hypertune
        self.verbose      = not if_quiet
        self.device       = device
        
        # model and optimizer are instantiate inside reconstruct() and hypertune()
        self.init_initializer()
        self.init_loss()
        self.init_constraint()
        print("\n### Done initializing PtyRADSolver ###")
    
    def init_initializer(self):
        print("\n### Initializing Initializer ###")
        self.init          = Initializer(self.params['exp_params'], self.params['source_params']).init_all()

    def init_loss(self):
        print("\n### Initializing loss function ###")
        self.loss_fn       = CombinedLoss(self.params['loss_params'], device=self.device)

    def init_constraint(self):
        print("\n### Initializing constraint function ###")
        self.constraint_fn = CombinedConstraint(self.params['constraint_params'], device=self.device, verbose=self.verbose)
    
    def reconstruct(self):
        params = self.params
        device = self.device

        # Create the model and optimizer, prepare indices, batches, and output_path
        model         = PtychoAD(self.init.init_variables, params['model_params'], device=device, verbose=self.verbose)
        optimizer     = torch.optim.Adam(model.optimizer_params)
        indices, batches, output_path = prepare_recon(model, self.init, params)
        recon_loop(model, self.init, params, optimizer, self.loss_fn, self.constraint_fn, indices, batches, output_path)
    
    def hypertune(self):
        import optuna
        hypertune_params = self.params['hypertune_params']
        n_trials         = hypertune_params.get('n_trials')
        study_name       = hypertune_params.get('study_name')
        storage_path     = hypertune_params.get('storage_path')
        pruner           = optuna.pruners.HyperbandPruner() if hypertune_params.get('use_pruning') else None

        # Create a study object and optimize the objective function
        study = optuna.create_study(
                    direction='minimize',
                    pruner=pruner,
                    storage=storage_path,  # Specify the storage URL here.
                    study_name=study_name,
                    load_if_exists=True)
        
        study.optimize(lambda trial: optuna_objective(trial, self.params, self.init, self.loss_fn, self.constraint_fn, self.device, self.verbose), n_trials=n_trials)
        print("Best hypertune params:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")
        
    def run(self):
        ''' Wrapper method for either regular reconstruction or hyperparameter tuning '''
        start_t = time_sync()
        solver_mode = 'hypertune' if self.if_hypertune else 'reconstruct'
        print(f"\n### Starting the PtyRADSolver in {solver_mode} mode ###")
        if self.if_hypertune:
            self.hypertune()
        else:
            self.reconstruct()
        end_t = time_sync()
        print(f"\n### The PtyRADSolver is finished in {end_t - start_t:.3f} sec ###")

def prepare_recon(model, init, params):
    
    vprint("\n### Generating indices, batches, and output_path ###", verbose=model.verbose)
    # Parse the variables
    init_variables = init.init_variables
    exp_params = params.get('exp_params')
    loss_params = params.get('loss_params')
    constraint_params = params.get('constraint_params')
    recon_params = params.get('recon_params')
    INDICES_MODE = recon_params['INDICES_MODE']
    GROUP_MODE = recon_params['GROUP_MODE']
    BATCH_SIZE = recon_params['BATCH_SIZE']
    output_dir = recon_params['output_dir']
    
    # Generate the indices, batches, output_path
    pos          = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    probe_int    = model.opt_probe.abs().pow(2).sum(0).detach().cpu().numpy()
    dx           = init_variables['dx']
    d_out        = get_blob_size(dx, probe_int, output='d90', verbose=model.verbose) # d_out unit is in Ang
    indices      = select_scan_indices(init_variables['N_scan_slow'], init_variables['N_scan_fast'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE, verbose=model.verbose)
    batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE, verbose=model.verbose)
    output_path  = make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, loss_params, verbose=model.verbose)

    fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
    fig_grouping.savefig(output_path + f"/summary_pos_grouping.png")
    return indices, batches, output_path

def recon_loop(model, init, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path):
    ''' Wrapper function for the optimization loop '''
    
    # Parse the variables
    init_variables    = init.init_variables
    exp_params        = params.get('exp_params')
    source_params     = params.get('source_params')
    loss_params       = params.get('loss_params')
    constraint_params = params.get('constraint_params')
    recon_params      = params.get('recon_params')
    NITER             = recon_params['NITER']
    SAVE_ITERS        = recon_params['SAVE_ITERS']
    fig_list          = recon_params['fig_list']
    
    vprint("\n### Start the PtyRAD iterative ptycho reconstruction ###", verbose=model.verbose)
    
    # Optimization loop
    loss_iters = []
    for niter in range(1,NITER+1):
        
        shuffle(batches)
        batch_losses, iter_t = recon_step(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=model.verbose)
        loss_iters.append((niter, loss_logger(batch_losses, niter, iter_t, verbose=model.verbose)))
        
        ## Saving intermediate results
        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            # Note that `exp_params` stores the initial exp_params, while `model` contains the actual params that could be updated if either meas_crop or meas_resample is not None
            save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses)
            
            ## Saving summary
            plot_summary(output_path, loss_iters, niter, indices, init_variables, model, fig_list, show_fig=False, save_fig=True, verbose=model.verbose)
    return loss_iters

def recon_step(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=True):
    ''' Perform 1 iteration/step of the ptycho reconstruciton in the optimization loop '''
    batch_losses = {name: [] for name in loss_fn.loss_params.keys()}
    start_iter_t = time_sync()
    
    # Start mini-batch optimization
    for batch_idx, batch in enumerate(batches):
        start_batch_t = time_sync()
        optimizer.zero_grad()
        model_DP, object_patches = model(batch)
        measured_DP = model.get_measurements(batch)
        loss_batch, losses = loss_fn(model_DP, measured_DP, object_patches, model.omode_occu)
        loss_batch.backward()
        optimizer.step() # batch update
        batch_t = time_sync() - start_batch_t

        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            batch_losses[loss_name].append(loss_value.detach().cpu().numpy())

        if batch_idx in np.linspace(0, len(batches)-1, num=6, dtype=int) and verbose:
            print(f"Done batch {batch_idx+1} in {batch_t:.3f} sec")
    
    # Apply iter-wise constraint
    constraint_fn(model, niter)
    
    iter_t = time_sync() - start_iter_t
    return batch_losses, iter_t

def loss_logger(batch_losses, niter, iter_t, verbose=True):
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    vprint(f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, "
          f"in {iter_t // 60} min {iter_t % 60:03f} sec\n", verbose=verbose)
    loss_iter = sum(avg_losses.values())
    return loss_iter

def optuna_objective(trial, params, init, loss_fn, constraint_fn, device='cuda:0', verbose=False):
    # Make this an independent function for the detailed walkthrough
    import optuna
    
    init.verbose = False
    params = copy.deepcopy(params)
        
    # Parse the variables
    exp_params        = params.get('exp_params')
    source_params     = params.get('source_params')
    loss_params       = params.get('loss_params')
    constraint_params = params.get('constraint_params')
    recon_params      = params.get('recon_params')
    NITER             = recon_params['NITER']
    SAVE_ITERS        = recon_params['SAVE_ITERS']
    fig_list          = recon_params['fig_list']
    
    # Parse the hypertune_params
    hypertune_params  = params['hypertune_params']
    tune_params       = hypertune_params['tune_params']
    trial_id = 't' + str(trial.number).zfill(4)
    params['recon_params']['prefix'] += trial_id
    
    # z_distance
    if tune_params['z_distance']['state']:
        z_distance_params = tune_params['z_distance']
        vmin, vmax, step = z_distance_params['min'], z_distance_params['max'], z_distance_params['step']
        z_distance = trial.suggest_float('z_distance', vmin, vmax, step=step)
        init.init_params['exp_params']['z_distance'] = z_distance
        init.init_H()
    
    # scan_affine
    scan_affine = []
    default_affine = {'scale':1, 'asymmetry':0, 'rotation':0, 'shear':0}
    for vname in ['scale', 'asymmetry', 'rotation', 'shear']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            vmin, vmax, step = vparams['min'], vparams['max'], vparams['step']
            scan_affine.append(trial.suggest_float(vname, vmin, vmax, step=step))
        else:
            scan_affine.append(default_affine[vname])
    if scan_affine != [1,0,0,0]:
        init.init_params['exp_params']['scan_affine'] = scan_affine
        init.init_pos()
        init.init_obj() # Update obj initialization because the scan range has changed
    
    # tilt (This will override the current tilts and force it to be a global tilt (2,1))
    obj_tilts = []
    for vname in ['tilt_y', 'tilt_x']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            vmin, vmax, step = vparams['min'], vparams['max'], vparams['step']
            obj_tilts.append(trial.suggest_float(vname, vmin, vmax, step=step))
        else:
            obj_tilts.append(0)
    obj_tilts = [obj_tilts] # Make it into [[tilt_y, tilt_x]]
    if obj_tilts != [[0,0]]:
        init.init_variables['obj_tilts'] = obj_tilts

    # Create the model and optimizer, prepare indices, batches, and output_path
    model         = PtychoAD(init.init_variables, params['model_params'], device=device, verbose=verbose)
    optimizer     = torch.optim.Adam(model.optimizer_params)
    indices, batches, output_path = prepare_recon(model, init, params)
    
    # Optimization loop
    loss_iters = []
    for niter in range(1, NITER+1):
        
        shuffle(batches)
        batch_losses, iter_t = recon_step(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=False) 
        loss_iter = loss_logger(batch_losses, niter, iter_t, verbose=False)
        loss_iters.append((niter, loss_iter))
        
        ## Saving intermediate results
        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            # Note that `exp_params` stores the initial exp_params, while `model` contains the actual params that could be updated if either meas_crop or meas_resample is not None
            save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses)
            
            ## Saving summary
            plot_summary(output_path, loss_iters, niter, indices, init.init_variables, model, fig_list, show_fig=False, save_fig=True, verbose=model.verbose)
        
        ## Pruning logic for optuna
        if hypertune_params['use_pruning']:
            trial.report(loss_iter, niter)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return loss_iter