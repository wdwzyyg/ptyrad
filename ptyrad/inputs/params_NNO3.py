# NNO3

ptycho_output_path = 'data/20240417_NNO3PlanviewPtychoRT_Lopa/Niter1000_Yi.mat' #Niter2000.mat
exp_CBED_path      = 'data/20240417_NNO3PlanviewPtychoRT_Lopa/centered.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 20.804, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk
    'dx_spec'           : 0.1848,# Ang, used to calculate dk
    'defocus'           : -130.928, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 4, #15, # Ang
    'Nlayer'            : 25, #10,
    'N_scans'           : 4096,
    'N_scan_slow'       : 64,
    'N_scan_fast'       : 64,
    'scan_step_size'    : 0.4398, # Ang
    'scan_flipT'        : (0,1,0), # (0,1,0) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'obj_tilts'         : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each', or 'load_PtyRAD'
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 12,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (3,2,0,1),
    'cbeds_reshape'     : (4096,128,128),
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': [exp_CBED_path, 'cbed'],
    'obj_source'         : 'simu',
    'obj_params'         : None,
    'probe_source'       : 'simu',
    'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None,
    # 'obj_source'         : 'PtyRAD', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyRAD',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyRAD',
    # 'pos_params'         : ptycho_output_path,
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}