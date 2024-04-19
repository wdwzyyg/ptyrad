# Si_128

ptycho_output_path = 'data/20240417_20231025_06-Si-pn5_Shake/Niter60.mat'
exp_CBED_path      = 'data/20240417_20231025_06-Si-pn5_Shake/06_df+25nm_7.2Mx_et100us_100pA_stable.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 21.198, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.2249,# Ang
    'defocus'           : -183.351, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
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
