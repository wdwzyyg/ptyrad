# multi_obj

ptycho_output_path = 'data/20240328_multi_object/cbed_WSe2_and_Cu_fov_42p86_39p97_dp180/1/roi_1_Ndp_180/MLs_L1_p2_g256_dpFlip_T/Niter100.mat'
exp_CBED_path      = 'data/20240328_multi_object/cbed_WSe2_and_Cu_fov_42p86_39p97_dp180/1/data_roi_1_Ndp_180_dp.hdf5' 

exp_params = {
    'kv'                : 80,  # kV
    'conv_angle'        : 20, # mrad, semi-convergence angle
    'Npix'              : 180, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk
    'dx_spec'           : 0.12,# Ang, used to calculate dk
    'defocus'           : -200, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention, which is opposite from Dr.Probe or TFS UI display
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 1,
    'N_scans'           : 9494,
    'N_scan_slow'       : 101,
    'N_scan_fast'       : 94,
    'scan_step_size'    : 0.3, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 2, #1
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 1, #2
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : None,
    'cbeds_reshape'     : None,
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
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