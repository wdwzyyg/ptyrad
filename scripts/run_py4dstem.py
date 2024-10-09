# ### py4DSTEM ptychography demo
# 
# 
# Last updated: 20240909 - dm852

import sys
print(sys.executable)
print(sys.version)
print(sys.version_info)

import py4DSTEM
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import tifffile
import time

py4DSTEM.__version__

work_dir = "/home/fs01/dm852/workspace/ptyrad_paper"
os.chdir(work_dir)
print("Current working dir: ", os.getcwd())

#######################
### start the clock ###
#######################
starttime = time.time()

## Define the save function for probe, object, scan_positions
def save_py4DSTEM(ptycho, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(save_dir + '.h5', "w") as f:
        f.create_group('output')
        f['output'].create_dataset('probe', data=ptycho.probe)
        f['output'].create_dataset('object', data=ptycho.object)
        f['output'].create_dataset('probe_pos', data=ptycho.positions)
    f.close()

def get_date(date_format = '%Y%m%d'):
    from datetime import date
    date_format = date_format
    date_str = date.today().strftime(date_format)
    return date_str

# Wrapper to intialize ptycho class
def initialize_ptycho(exp_params):
    if exp_params['Nlayer'] == 1:
        ms_ptycho = py4DSTEM.process.phase.MixedstatePtychography(
            datacube=dataset,
            num_probes = exp_params['pmode_max'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            device='gpu', 
            storage='cpu', 
        )
    else:
        ms_ptycho = py4DSTEM.process.phase.MixedstateMultislicePtychography(
            datacube=dataset,
            num_probes = exp_params['pmode_max'],
            num_slices=exp_params['Nlayer'],
            slice_thicknesses=exp_params['z_distance'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            device='gpu', 
            storage='cpu',
        )
    return ms_ptycho


# Note: py4dstem always estimates an affine transformation instead of taking user input
exp_params = {
    'kv'                : 80, # type: float, unit: kV. Acceleration voltage for relativistic electron wavelength calculation
    'conv_angle'        : 24.9, # type: float, unit: mrad. Semi-convergence angle for probe-forming aperture
    'Npix'              : 128, # type: integer, unit: px (k-space). Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'dx_spec'           : 0.1494, # type: float, unit: Ang. Real space pixel size calibration at specimen plane (object, probe, and probe positions share the same pixel size)
    'defocus'           : 0, # type: float, unit: Ang. Defocus (-C1) aberration coefficient for the probe. Positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # type: float, unit: Ang. Spherical aberration coefficient (Cs) for the probe
    'z_distance'        : 2, # type: float, unit: Ang. Slice thickness for multislice ptychography. Typical values are between 1 to 20 Ang
    'Nlayer'            : 6, # type: int, unit: #. Number of slices for multislice object
    'N_scans'           : 16384, # type: int, unit: #. Number of probe positions (or equivalently diffraction patterns since 1 DP / position)
    'N_scan_slow'       : 128, # type: int, unit: #. Number of scan position along slow scan direction. Usually it's the vertical direction of acquisition GUI
    'N_scan_fast'       : 128, # type: int, unit: #. Number of scan position along fast scan direction. usually it's the horizontal direction of acquisition GUI
    'scan_step_size'    : 0.4290, # type: float, unit: Ang. Step size between probe positions in a rectangular raster scan pattern
    'scan_flipT'        : None, # type: None or list of 3 binary booleans (0 or 1) as [flipup, fliplr, transpose] just like PtychoShleves. Default value is None or equivalently [0,0,0]. This applies additional flip and transpose to initialized scan patterns. Note that modifing 'scan_flipT' would change the image orientation, so it's recommended to set this to None, and only use 'meas_flipT' to get the orientation correct
    'scan_affine'       : None, # type: None or list of 4 floats as [scale, asymmetry, rotation, shear] just like PtychoShleves. Default is None or equivalently [1,0,0,0], rotation and shear are in unit of degree. This applies additional affine transformation to initialized scan patterns to correct sample drift and imperfect scan coils
    'pmode_max'         : 6, # type: int, unit: #. Maximum number of mixed probe modes. Set to pmode_max = 1 for single probe state, pmode_max > 1 for mixed-state probe during initialization. For simulated initial object, it'll be generated with the specified number of probe modes. For loaded probe, the pmode dimension would be capped at this number
}

recon_params = {
    'NITER': 64, # type: int. Total number of reconstruction iterations. 1 iteration means a full pass of all selected diffraction patterns. Usually 20-50 iterations can get 90% of the work done with a proper learning rate between 1e-3 to 1e-4. For faster trials in hypertune mode, set 'NITER' to a smaller number than your typical reconstruction to save time. Usually 10-20 iterations are enough for the hypertune parameters to show their relative performance. 
    'BATCH_SIZE': 32, # type: int. Number of diffraction patterns processed simultaneously to get the gradient update. "Batch size" is commonly used in machine learning community, while it's called "grouping" in PtychoShelves. Batch size has an effect on both convergence speed and final quality, usually smaller batch size leads to better final quality for iterative gradient descent, but smaller batch size would also lead to longer computation time per iteration because the GPU isn't as utilized as large batch sizes (due to less GPU parallelism). Generally batch size of 32 to 128 is used, although certain algorithms (like ePIE) would prefer a large batch size that is equal to the dataset size for robustness. For extremely large object (or with a lot of object modes), you'll need to reduce batch_size to save GPU memory as well.
    'step_size': 0.8,
    'output_dir': 'publication/output/tBL-WSe2/', # type str. Path and name of the main output directory. Ideally the 'output_dir' keeps a series of reconstruction of the same materials system or project. The PtyRAD results and figs will be saved into a reconstruction-specific folder under 'output_dir'. The 'output_dir' folder will be automatically created if it doesn't exist.
    'prefix_date': True, # type: boolean. Whether to prefix a date str to the reconstruction folder or not. Set to true to automatically prefix a date str like '20240903_' in front of the reconstruction folder name. Suggested value is true for both reconstruction and hypertune modes. In hypertune mode, the date string would be applied on the hypertune folder instead of the reconsstruction folder. 
    'prefix': '', # type: str. Prefix this string to the reconstruction folder name. Note that "_" will be automatically generated, and the attached str would be after the date str if 'prefix_date' is true. In hypertune mode, the prefix string would be applied on the hypertune folder instead of the reconsstruction folder.  
    'postfix': '', # type: str. Postfix this string to the reconstruction folder name. Note that "_" will be automatically generated. In hypertune mode, the postfix string would be applied on the hypertune folder instead of the reconsstruction folder.  
}



file_path = '/media/muller_group/dm852_ExPro/ptyrad_paper/'
file_data = file_path + 'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/data_roi1_Ndp128_step128_dp.hdf5'
with h5py.File(file_data, 'r') as f:
    dataset = py4DSTEM.DataCube(np.reshape(f['dp'],[exp_params['Npix'],exp_params['Npix'],exp_params['N_scan_slow'],exp_params['N_scan_fast']]))

dataset.calibration.set_R_pixel_size(exp_params['scan_step_size']/10)
dataset.calibration.set_R_pixel_units('nm')
dataset.calibration.set_Q_pixel_size(1/(exp_params['dx_spec']*exp_params['Npix']))
dataset.calibration.set_Q_pixel_units('A^-1')

ptycho = initialize_ptycho(exp_params)
ptycho.preprocess(
    plot_center_of_mass = False,
    plot_rotation=False,
    # plot_probe_overlaps=False,
    # force_com_rotation = -87.0,
)

ptycho.reconstruct(
    reset=True,
    store_iterations=True,
    num_iter = recon_params['NITER'],
    normalization_min=1,
    step_size = recon_params['step_size'],
    #gaussian_filter_sigma = .5,
    #kz_regularization_gamma = 1000,
    #reconstruction_method = 'RAAR'
    max_batch_size = recon_params['BATCH_SIZE'],
).visualize(
    iterations_grid = 'auto'
)


# Set save path
prefix = recon_params['prefix']
prefix  = prefix+ '_' if prefix  != '' else '' + 'py4DSTEM' + '_'
prefix = get_date() + '_' + prefix 

output_dir = recon_params['output_dir']
save_dir  = file_path + output_dir + prefix + f"N{(exp_params['N_scan_slow']*exp_params['N_scan_fast'])}_dp{exp_params['Npix']}" 

# Save probe, object and scan_positions to .h5 file
save_py4DSTEM(ptycho, save_dir)

#####################
### end the clock ###
#####################
endtime = time.time()

#####################
### Print summary ###
#####################
print('Printing summary')
print('Loaded measurement data from:', file_data)
print('4D data shape:', dataset.shape)
print('Ptycho engine:', ms_ptycho.__class__)
print("Saved results to:", save_dir+'.h5')
print('probe shape:', ms_ptycho.probe.shape)
print('object shape:', ms_ptycho.object.shape)
print('Total iterations:', recon_params['NITER'])
print('Total time comsumption:', endtime - starttime)