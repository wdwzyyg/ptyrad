# Si_128

ptycho_output_path = 'H:/workspace/p01_code/deep_ptycho/ptyrad/data/20240417_20231025_06-Si-pn5_Shake/Niter60.mat'
exp_CBED_path      = 'H:/workspace/p01_code/deep_ptycho/ptyrad/data/20240417_20231025_06-Si-pn5_Shake/06_df+25nm_7.2Mx_et100us_100pA_stable.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 21.198, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.2249,# Ang
    'defocus'           : 0, #-183.351, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 42,
    'N_scans'           : 65536,
    'N_scan_slow'       : 256,
    'N_scan_fast'       : 256,
    'scan_step_size'    : 0.44, # Ang
    'scan_flip'         : None, # (1) for 'simu' pos, None for loaded pos
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1,
    'pmode_max'         : 5,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,1,3,2),
    'cbeds_reshape'     : (65536,128,128),
    'cbeds_flip'        : None,
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': [exp_CBED_path, 'cbed'],
    'obj_source'         : 'simu', #'simu',
    'obj_params'         : None, #ptycho_output_path, #(1,44,860,860)
    'probe_source'       : 'simu',
    'probe_params'       : None, 
    # 'pos_source'         : 'simu',
    # 'pos_params'         : None, 
    # 'obj_source'         : 'PtyShv', #'PtyShv',
    # 'obj_params'         : ptycho_output_path, #ptycho_output_path, #(1,8,391,403),
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    'pos_source'         : 'PtyShv',
    'pos_params'         : ptycho_output_path,
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}

model_params = {
    'detector_blur_std': None,
    'lr_params':{
        'obja': 5e-4,
        'objp': 5e-4,
        'probe': 1e-3, 
        'probe_pos_shifts': 0}}

loss_params = {
    'loss_single': {'state':  True,  'weight': 1.0, 'dp_pow': 0.5},
    'loss_pacbed': {'state': False,  'weight': 1.0, 'dp_pow': 0.2},
    'loss_tv'    : {'state': False,  'weight': 1e-5},
    'loss_l1'    : {'state': False,  'weight': 0.1},
    'loss_l2'    : {'state': False,  'weight': 1.0},
    'loss_postiv': {'state': False,  'weight': 1.0}
}

constraint_params = {
    'ortho_pmode'   : {'freq': None},
    'ortho_omode'   : {'freq': None},
    'kz_filter'     : {'freq': 1, 'beta':1, 'alpha':1, 'z_pad':None},
    'postiv'        : {'freq': 1},
    'fix_probe_int' : {'freq': 1}
}

# Reconstruction params
NITER        = 20
INDICES_MODE = 'center'   # 'full', 'center', 'sub'
BATCH_SIZE   = 64
GROUP_MODE   = 'sparse' # 'random', 'sparse', 'compact'
SAVE_ITERS   = 5        # scalar or None

output_dir   = 'H:/workspace/p01_code/deep_ptycho/ptyrad/output/Si'
postfix      = '_focusProbe_noSVD'