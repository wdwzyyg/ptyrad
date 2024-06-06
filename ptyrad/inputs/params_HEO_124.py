# HEO_124

ptycho_output_path = '/home/muller_group/Documents/pytcho_dm852/ptyrad/data/20240520_HEO_data21/Niter1000.mat'
exp_CBED_path      = 'data/20240529_HEOnp_Desheng/data_roi0_Ndp124_dp.hdf5'

exp_params = {
    'kv'                : 200,  # kV
    'conv_angle'        : 30.0, # mrad, semi-convergence angle
    'Npix'              : 124, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.2117, #0.12845 for 10Mx,#0.2117,# Ang dx = 1/(dk*Npix)
    'defocus'           : -144.0, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 15,
    'N_scans'           : 128*128,
    'N_scan_slow'       : 128,
    'N_scan_fast'       : 128,
    'scan_step_size'    : 0.5121, # 0.536, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,2,1),
    'cbeds_reshape'     : None,
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5', #'hdf5'
    'measurements_params': {'path': exp_CBED_path, 'key':'dp'}, #'dp', '/datacube_root/datacube/data'
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