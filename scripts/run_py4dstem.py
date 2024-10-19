# Python script to run py4DSTEM
# Updated by Chia-Hao Lee on 2024.10.09

import argparse
import sys
import py4DSTEM
import numpy as np
import h5py
import os
from time import time

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r") as file:
        params_dict = yaml.safe_load(file)
    print("Success! Loaded .yml file path =", file_path)
    params_dict['params_path'] = file_path
    return params_dict

def initialize_ptycho(datacube, exp_params):
    
    Npix = exp_params['Npix']
    N_scan_fast = exp_params['N_scan_fast']
    N_scan_slow = exp_params['N_scan_slow']
    dx_spec = exp_params['dx_spec']
    scan_step_size = exp_params['scan_step_size']
    
    pos_extent = np.array([N_scan_slow,N_scan_fast]) * scan_step_size / dx_spec
    object_extent = 1.2 * (pos_extent + Npix)
    object_padding_px = tuple((object_extent - pos_extent)//2)
    
    if exp_params['Nlayer'] == 1:
        print("Initializing MixedstatePtychography")
        ptycho = py4DSTEM.process.phase.MixedstatePtychography(
            datacube=datacube,
            num_probes = exp_params['pmode_max'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            object_padding_px = object_padding_px,
            device='gpu', 
            storage='cpu', 
        )
    else:
        print("Initializing MixedstateMultislicePtychography")
        ptycho = py4DSTEM.process.phase.MixedstateMultislicePtychography(
            datacube=datacube,
            num_probes = exp_params['pmode_max'],
            num_slices=exp_params['Nlayer'],
            slice_thicknesses=exp_params['z_distance'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            object_padding_px = object_padding_px,
            device='gpu', 
            storage='cpu',
        )
    return ptycho

def get_datacube(exp_params):
    
    data_path = exp_params['measurements_params'].get('path')
    data_key  = exp_params['measurements_params'].get('key')
    
    with h5py.File(data_path, 'r') as f:
        meas = np.array(f[data_key])
        
        # Flip the measurements
        flipT_axes = exp_params['meas_flipT']
        print(f"Flipping measurements with [flipup, fliplr, transpose] = {flipT_axes}")
        if flipT_axes[0] != 0:
            meas = np.flip(meas, 1)
        if flipT_axes[1] != 0:
            meas = np.flip(meas, 2)
        if flipT_axes[2] != 0:
            meas = np.transpose(meas, (0,2,1))
        
        # Reshape
        dataset = np.reshape(meas, [exp_params['N_scan_slow'],exp_params['N_scan_fast'],exp_params['Npix'],exp_params['Npix']])
        
        # Calibrate py4DSTEM datacube
        datacube = py4DSTEM.DataCube(dataset)
        datacube.calibration.set_R_pixel_size(exp_params['scan_step_size'])
        datacube.calibration.set_R_pixel_units('A')
        datacube.calibration.set_Q_pixel_size(1/(exp_params['dx_spec']*exp_params['Npix']))
        datacube.calibration.set_Q_pixel_units('A^-1')
    return datacube

def get_date(date_format = '%Y%m%d'):
    from datetime import date
    date_format = date_format
    date_str = date.today().strftime(date_format)
    return date_str

def make_output_folder(params):
    exp_params = params['exp_params']
    recon_params = params['recon_params']

    # Preprocess prefix and postfix
    prefix = recon_params['prefix']
    postfix = recon_params['postfix']
    prefix = prefix + '_' if prefix  != '' else ''
    postfix = '_'+ postfix if postfix != '' else ''

    if recon_params['prefix_date']:
        prefix = get_date() + '_' + prefix 

    # Append basic parameters to folder name
    output_dir  = recon_params['output_dir']
    meas_flipT  = exp_params['meas_flipT'] 
    folder_str = prefix + f"N{(exp_params['N_scans'])}_dp{exp_params['Npix']}"

    if meas_flipT is not None:
        folder_str = folder_str + '_flipT' + ''.join(str(x) for x in meas_flipT)

    folder_str += f"_random{recon_params['BATCH_SIZE']}_p{exp_params['pmode_max']}_{exp_params['Nlayer']}slice"

    if exp_params['Nlayer'] != 1:
        z_distance = np.array(exp_params['z_distance']).round(2)
        folder_str += f"_dz{z_distance:.3g}"

    output_path = os.path.join(output_dir, folder_str)
    output_path += postfix
    os.makedirs(output_path, exist_ok=True)
    print(f"output_path = '{output_path}' is generated!")
    return output_path

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run py4DSTEM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid",       type=int, required=False, default=None)
    args = parser.parse_args()
    
    ## Load params and initialize datacube
    params       = load_yml_params(args.params_path)
    exp_params   = params['exp_params']
    recon_params = params['recon_params']
    datacube     = get_datacube(exp_params)
    
    ## Initialize py4dstem ptycho instance
    ptycho = initialize_ptycho(datacube, exp_params)
    ptycho.preprocess(
        plot_center_of_mass = False,
        plot_rotation=False,
    )
    
    ## Reconstruct py4dstem ptycho
    output_path = make_output_folder(params)
    
    solver_start_t = time()
    
    ptycho.reconstruct(
    num_iter = recon_params['NITER'],
    reconstruction_method = 'gradient-descent',
    max_batch_size = recon_params['BATCH_SIZE'],
    step_size = 0.1, # Update step size, default is 0.5 but 0.1 is numerically more stable for multislice
    kz_regularization_gamma = 1,
    object_positivity = False,
    reset = True, # If True, previous reconstructions are ignored
    progress_bar = False, # If True, reconstruction progress is displayed
    store_iterations = False, # If True, reconstructed objects and probes are stored at each iteration.
    save_iters = recon_params['SAVE_ITERS'], # Added by CHL to save intermediate results
    output_path = output_path)
    
    solver_end_t = time()
    print(f"py4DSTEM ptycho solver is finished in {parse_sec_to_time_str(solver_end_t - solver_start_t)}")