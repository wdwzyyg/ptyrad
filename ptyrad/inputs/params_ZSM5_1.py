# ZSM5-1

ptycho_output_path = 'data/20240812_Zeolite_Guanxing/Dataset_1/Niter200.mat'
exp_CBED_path      = 'data/20240812_Zeolite_Guanxing/Dataset_1/data_roi_full_dp.hdf5' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 14.279, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.3712,# Ang
    'defocus'           : 110, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 40, # Ang
    'Nlayer'            : 9,
    'N_scans'           : 60516,
    'N_scan_slow'       : 246,
    'N_scan_fast'       : 246,
    'scan_step_size'    : 0.40853, # Ang
    'scan_flipT'        : (0,1,0), # (0,1,0) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, #(1,0,2,0), # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : None, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'meas_permute'           : None,
    'meas_reshape'           : None,
    'meas_flipT'             : [0,0,1], # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'meas_crop'              : [[0,246],[0,246],[2,126],[0,124]], # None or (4,2) array-like for [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]
    'meas_resample'          : None, # None or (2,1) array-like for [ky_zoom, kx_zoom]
    'meas_add_source_size'   : None, # None or a scalar of std (Ang)
    'meas_add_detector_blur' : None, # None or a scalar of std (px)
    'meas_add_poisson_noise' : None, # None or a scalar of electrons/Ang^2
    'probe_simu_params'      : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': {'path': exp_CBED_path, 'key':'dp'},
    'obj_source'         : 'simu', 
    'obj_params'         : None,
    'probe_source'       : 'simu',
    'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None,
    'tilt_source'        : 'simu',
    'tilt_params'        : {'tilt_type':'all', 'init_tilts':[[0,6.74]]}, # 'init_tilts' = (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each' 
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