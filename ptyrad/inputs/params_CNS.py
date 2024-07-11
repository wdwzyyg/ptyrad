# CNS

ptycho_output_path = 'data/CNS_from_Hari/Niter10000.mat'
exp_CBED_path      = 'data/CNS_from_Hari/231002_fov_23p044A_x_24p402A_thickness_9p978A_step0p28_conv30_dfm100_det70_TDS_500configs_xdirection_Co_0p25_Nb_0_S_0.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 30, # mrad, semi-convergence angle
    'Npix'              : 164, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk
    'dx_spec'           : 0.1406,# Ang, used to calculate dk
    'defocus'           : -100, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention, which is opposite from Dr.Probe or TFS UI display
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 1,
    'N_scans'           : 7134,
    'N_scan_slow'       : 87,
    'N_scan_fast'       : 82,
    'scan_step_size'    : 0.28, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1, #1
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 2, #2
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'meas_permute'           : (0,1,3,2),
    'meas_reshape'           : (7134,164,164),
    'meas_flipT'             : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'meas_crop'              : None, # None or (4,2) array-like for [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]
    'meas_resample'          : None, # None or (2,1) array-like for [ky_zoom, kx_zoom]
    'meas_add_source_size'   : None, # None or a scalar of std (Ang)
    'meas_add_detector_blur' : None, # None or a scalar of std (px)
    'meas_add_poisson_noise' : None, # None or a scalar of electrons/Ang^2
    'probe_simu_params'      : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': {'path': exp_CBED_path, 'key':'cbed'},
    'obj_source'         : 'simu',
    'obj_params'         : None,
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