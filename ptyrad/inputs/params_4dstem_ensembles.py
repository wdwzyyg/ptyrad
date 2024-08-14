# pymatgen

# ptycho_output_path = 'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/roi1_Ndp128_step128/MLs_L1_p10_g128_pc0_noModel_updW100_mm_dpFlip_ud_T/Niter9000_v7.mat'
exp_CBED_path      = 'H:/workspace/ptyrad/data/20240719_pymatgen_4dstem/static_coherent/mp-149_80kV_alpha40_z21.hdf5'
# exp_CBED_path      = 'H:/workspace/ptyrad/data/20240719_pymatgen_4dstem/phonon_temporal/mp-149_80kV_alpha40_z21.hdf5'

exp_params = {
    'kv'                : 80,  # kV
    'conv_angle'        : 40, # mrad, semi-convergence angle
    'Npix'              : 583, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.075,# Ang
    'defocus'           : -200, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 500, # Ang, spherical aberration coefficients
    'z_distance'        : 1, # Ang
    'Nlayer'            : 21,
    'N_scans'           : 1681,
    'N_scan_slow'       : 41,
    'N_scan_fast'       : 41,
    'scan_step_size'    : 0.50, # Ang
    'scan_flipT'        : None, # None for both 'simu' and loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : None, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 1,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'meas_permute'      : None,
    'meas_reshape'      : [1681,583,583],
    'meas_flipT'        : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'meas_crop'              : None, # None or (4,2) array-like for [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]
    'meas_resample'          : [384/583,384/583], # None or (2,1) array-like for [ky_zoom, kx_zoom]
    'meas_add_source_size'   : None, # None or a scalar of std (Ang)
    'meas_add_detector_blur' : None, # None or a scalar of std (px)
    'meas_add_poisson_noise' : None, # None or a scalar of electrons/Ang^2
    'probe_simu_params'      : None
    }

from tifffile import imread
import numpy as np
obj = imread('data/20240719_pymatgen_4dstem/static_coherent/gt_phase_mp_149_80kV_z21_canvas.tif')[None,]
obj = np.exp(1j * obj)

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': {'path': exp_CBED_path, 'key':'/dp'},
    'obj_source'         : 'simu', #'custom',
    'obj_params'         : None, #obj,
    'probe_source'       : 'simu',
    'probe_params'       : None,  
    'pos_source'         : 'simu',
    'pos_params'         : None,
    'tilt_source'        : 'simu',
    'tilt_params'        : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # 'init_tilts' = (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each' 
    # 'obj_source'         : 'PtyRAD', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyRAD',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyRAD',
    # 'pos_params'         : ptycho_output_path,
    # 'tilt_source'        : 'PtyRAD',
    # 'tilt_params'        : 'ptycho_output_path,
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}