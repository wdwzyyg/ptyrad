## Define the ptycho reconstruction solver class and functions

import copy
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch

from ptyrad.initialization import Initializer
from ptyrad.models import PtychoAD
from ptyrad.optimization import CombinedConstraint, CombinedLoss
from ptyrad.utils import (
    get_blob_size,
    make_batches,
    make_output_folder,
    save_results,
    select_scan_indices,
    time_sync,
    vprint,
)
from ptyrad.visualization import plot_pos_grouping, plot_summary

class PtyRADSolver:
    """
    A wrapper class to perform ptychographic reconstruction or hyperparameter tuning.

    The PtyRADSolver class initializes the necessary components for ptychographic 
    reconstruction and provides methods to execute the reconstruction or perform 
    hyperparameter tuning using Optuna.

    Attributes:
        params (dict): Dictionary containing all the parameters required for 
            initialization, loss functions, constraints, model, and optional 
            hyperparameter tuning.
        if_hypertune (bool): A flag to indicate whether hyperparameter tuning should 
            be performed instead of regular reconstruction. Defaults to False.
        verbose (bool): A flag to control the verbosity of the output. Defaults to True unless
            if_quiet is set to True.
        device (str): The device to run the computations on (e.g., 'cuda:0' for GPU, 'cpu' for CPU). 
            Defaults to 'cuda:0'.

    Methods:
        init_initializer():
            Initializes the variables and objects needed for the reconstruction process.
        init_loss():
            Initializes the loss function using the provided parameters.
        init_constraint():
            Initializes the constraint function using the provided parameters.
        reconstruct():
            Executes the ptychographic reconstruction process by creating the model, 
            optimizer, and running the reconstruction loop.
        hypertune():
            Performs hyperparameter tuning using Optuna.
        run():
            A wrapper method to run the solver in either reconstruction or hyperparameter 
            tuning mode based on the if_hypertune flag.
    """
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
        pruner           = optuna.pruners.HyperbandPruner(min_resource=5, reduction_factor=2) if hypertune_params.get('use_pruning') else None

        # Create a study object and optimize the objective function
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True)
        study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    pruner=pruner,
                    storage=storage_path,  # Specify the storage URL here.
                    study_name=study_name,
                    load_if_exists=True)
        
        study.optimize(lambda trial: optuna_objective(trial, self.params, self.init, self.loss_fn, self.constraint_fn, self.device, self.verbose), n_trials=n_trials)
        print("Best hypertune params:")
        for key, value in study.best_params.items():
            print(f"\t{key}: {value}")
        
    def run(self):
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
    """
    Prepares the indices, batches, and output path for ptychographic reconstruction.

    This function parses the necessary parameters and generates the indices for scanning, 
    creates batches based on the probe positions, and sets up the output directory for 
    saving results. It also plots and saves a figure illustrating the grouping of probe 
    positions.

    Args:
        model (PtychoAD): The ptychographic model containing the object, probe, 
            probe positions, and other relevant parameters.
        init (Initializer): The initializer object containing the initialized variables 
            needed for reconstruction.
        params (dict): A dictionary containing various parameters needed for the 
            reconstruction process, including experimental parameters, loss parameters, 
            constraint parameters, and reconstruction settings.

    Returns:
        tuple: A tuple containing the following:
            - indices (numpy.ndarray): Array of indices for scanning positions.
            - batches (list of numpy.ndarray): List of batches where each batch contains 
              indices grouped according to the selected grouping mode.
            - output_path (str): The path to the directory where reconstruction results 
              and figures will be saved.
    """
    vprint("\n### Generating indices, batches, and output_path ###", verbose=model.verbose)
    # Parse the variables
    init_variables = init.init_variables
    exp_params = init.init_params.get('exp_params') # These could be modified by Optuna, hence can be different from params['exp_params]
    loss_params = params.get('loss_params')
    constraint_params = params.get('constraint_params')
    recon_params = params.get('recon_params')
    INDICES_MODE = recon_params['INDICES_MODE']
    subscan_slow = recon_params.get("subscan_slow")
    subscan_fast = recon_params.get("subscan_fast")
    GROUP_MODE = recon_params['GROUP_MODE']
    BATCH_SIZE = recon_params['BATCH_SIZE']
    output_dir = recon_params['output_dir']
    dir_affixes = recon_params['dir_affixes']
    
    # Generate the indices, batches, output_path
    pos          = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    probe_int    = model.opt_probe.abs().pow(2).sum(0).detach().cpu().numpy()
    dx           = init_variables['dx']
    d_out        = get_blob_size(dx, probe_int, output='d90', verbose=model.verbose) # d_out unit is in Ang
    indices      = select_scan_indices(init_variables['N_scan_slow'], init_variables['N_scan_fast'], subscan_slow=subscan_slow, subscan_fast=subscan_fast, mode=INDICES_MODE, verbose=model.verbose)
    batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE, verbose=model.verbose)
    output_path  = make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, loss_params, dir_affixes, verbose=model.verbose)

    fig_grouping = plot_pos_grouping(pos, batches, circle_diameter=d_out/dx, diameter_type='90%', dot_scale=1, show_fig=False, pass_fig=True)
    fig_grouping.savefig(output_path + "/summary_pos_grouping.png")
    plt.close(fig_grouping)
    return indices, batches, output_path

def recon_loop(model, init, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path):
    """
    Executes the iterative optimization loop for ptychographic reconstruction.

    This function performs the iterative reconstruction process by optimizing the model 
    parameters over a specified number of iterations. During each iteration, it applies 
    the loss and constraint functions, updates the model, and logs the loss values. 
    Intermediate results are saved at specified intervals, and a summary is plotted.

    Args:
        model (PtychoAD): The ptychographic model containing the parameters and variables 
            to be optimized.
        init (Initializer): The initializer object containing the initialized variables 
            needed for reconstruction.
        params (dict): A dictionary containing various parameters for the reconstruction 
            process, including experimental parameters, source parameters, loss parameters, 
            constraint parameters, and reconstruction settings.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (CombinedLoss): The loss function object used to compute the loss during 
            each iteration.
        constraint_fn (CombinedConstraint): The constraint function object applied during 
            each iteration to enforce specific constraints on the model.
        indices (numpy.ndarray): Array of indices for scanning positions.
        batches (list of numpy.ndarray): List of batches where each batch contains indices 
            grouped according to the selected grouping mode.
        output_path (str): The path to the directory where reconstruction results and 
            figures will be saved.

    Returns:
        list: A list of tuples, where each tuple contains the iteration number, the loss 
            value for that iteration, and the time taken for that iteration.
    """
    
    # Parse the variables
    init_variables    = init.init_variables
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
            save_results(output_path, model, params, loss_iters, iter_t, niter, batch_losses)
            
            ## Saving summary
            plot_summary(output_path, model, loss_iters, niter, indices, init_variables, fig_list=fig_list, show_fig=False, save_fig=True, verbose=model.verbose)
    return loss_iters

def recon_step(batches, model, optimizer, loss_fn, constraint_fn, niter, verbose=True):
    """
    Performs one iteration (or step) of the ptychographic reconstruction in the optimization loop.

    This function executes a single iteration of the reconstruction process, including:
    - Computing the forward model to generate diffraction patterns.
    - Calculating the loss by comparing the modeled and measured diffraction patterns.
    - Performing a backward pass to compute gradients and update the model parameters using the optimizer.
    - Applying iteration-wise constraints after all batches are processed.

    Args:
        batches (list of numpy.ndarray): List of batches where each batch contains indices 
            grouped according to the selected grouping mode.
        model (PtychoAD): The ptychographic model containing the parameters and variables 
            to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (CombinedLoss): The loss function object used to compute the loss for each batch.
        constraint_fn (CombinedConstraint): The constraint function object applied after each iteration 
            to enforce specific constraints on the model.
        niter (int): The current iteration number in the optimization loop.
        verbose (bool, optional): If True, prints progress information during the batch processing. 
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - batch_losses (dict): A dictionary where each key corresponds to a loss component name, 
              and the value is a list of loss values computed for each batch in the iteration.
            - iter_t (float): The total time taken to complete the iteration.
    """
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
    """
    Logs and summarizes the loss values for an iteration during the ptychographic reconstruction.

    This function computes the average loss for each loss component across all batches in the 
    current iteration. It then logs the total loss, the individual loss components, and the 
    time taken for the iteration. The function also returns the total loss for the iteration.

    Args:
        batch_losses (dict): A dictionary where each key corresponds to a loss component name, 
            and the value is a list of loss values computed for each batch in the iteration.
        niter (int): The current iteration number in the optimization loop.
        iter_t (float): The total time taken to complete the iteration, in seconds.
        verbose (bool, optional): If True, prints the loss summary to the console. Defaults to True.

    Returns:
        float: The total loss for the current iteration, computed as the sum of the average 
        loss values for each component.
    """
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    vprint(f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, "
          f"in {iter_t // 60} min {iter_t % 60:03f} sec\n", verbose=verbose)
    loss_iter = sum(avg_losses.values())
    return loss_iter

def optuna_objective(trial, params, init, loss_fn, constraint_fn, device='cuda:0', verbose=False):
    """
    Objective function for Optuna hyperparameter tuning in ptychographic reconstruction.

    This function is used by Optuna to optimize the hyperparameters of the ptychographic reconstruction
    process. The function updates the reconstruction parameters based on the trial's suggestions and 
    runs the reconstruction loop to evaluate the performance. The function also implements Optuna's 
    pruning mechanism to stop unpromising trials early.

    Args:
        trial (optuna.trial.Trial): A trial object that suggests hyperparameter values and handles 
            pruning.
        params (dict): A dictionary containing all the parameters for the reconstruction, including 
            experimental parameters, model parameters, and hyperparameter tuning configurations.
        init (Initializer): An instance of the Initializer class that holds initialized variables 
            and methods for updating them based on the trial's suggestions.
        loss_fn (CombinedLoss): The loss function object that calculates the reconstruction loss.
        constraint_fn (CombinedConstraint): The constraint function object that applies constraints 
            during optimization.
        device (str, optional): The device to run the reconstruction on, e.g., 'cuda:0'. Defaults to 'cuda:0'.
        verbose (bool, optional): If True, enables verbose output. Defaults to False.

    Returns:
        float: The total loss for the final iteration of the reconstruction process, used by Optuna 
        to evaluate the trial's performance.

    Raises:
        optuna.exceptions.TrialPruned: Raised when the trial should be pruned based on the 
        intermediate results.
    """
    import optuna
    
    init.verbose = False
    params = copy.deepcopy(params)
        
    # Parse the variables
    recon_params      = params.get('recon_params')
    NITER             = recon_params['NITER']
    SAVE_ITERS        = recon_params['SAVE_ITERS']
    fig_list          = recon_params['fig_list']
    
    # Parse the hypertune_params
    hypertune_params  = params['hypertune_params']
    tune_params       = hypertune_params['tune_params']
    trial_id = 't' + str(trial.number).zfill(4)
    params['recon_params']['prefix'] += trial_id
    
    ## Currently only re-initialize the required parts for performance, but once there're too many correlated params need to be re-initialized,
    ## we might put the entire initialization inside optuna_objective for readability, although init_measurements for every trial would be a large overhead.
    ## For example, re-initialize `dx_spec` would require re-initializing everything including the 4D-STEM data.
    
    # probe_params (conv_angle, defocus)
    remake_probe = False
    for vname in ['conv_angle', 'defocus']:
        if tune_params[vname]['state']:
            vparams = tune_params[vname]
            vmin, vmax, step = vparams['min'], vparams['max'], vparams['step']
            init.init_params['exp_params'][vname] = trial.suggest_float(vname, vmin, vmax, step=step)
            remake_probe = True
        if remake_probe:
            init.init_probe()
            
    # z_distance
    if tune_params['z_distance']['state']:
        z_distance_params = tune_params['z_distance']
        vmin, vmax, step = z_distance_params['min'], z_distance_params['max'], z_distance_params['step']
        z_distance = trial.suggest_float('z_distance', vmin, vmax, step=step)
        init.init_params['exp_params']['z_distance'] = z_distance
        init.init_H()
    
    # scan_affine
    scan_affine = []
    scan_affine_init = params['exp_params']['scan_affine']
    if scan_affine_init is not None:
        default_affine = {'scale':scan_affine_init[0], 'asymmetry':scan_affine_init[1], 'rotation':scan_affine_init[2], 'shear':scan_affine_init[3]}
    else:
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
            save_results(output_path, model, params, loss_iters, iter_t, niter, batch_losses)
            
            ## Saving summary
            plot_summary(output_path, model, loss_iters, niter, indices, init.init_variables, fig_list=fig_list, show_fig=False, save_fig=True, verbose=model.verbose)
        
        ## Pruning logic for optuna
        if hypertune_params['use_pruning']:
            trial.report(loss_iter, niter)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return loss_iter