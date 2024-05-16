# bSTO

ptycho_output_path = 'data/20240508_bSTO_Hari/Niter100.mat'
exp_CBED_path      = 'data/20240508_bSTO_Hari/01_crop198_256_x25_y1_centered.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 31.6, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk
    'dx_spec'           : 0.1752,# Ang, used to calculate dk
    'defocus'           : -110, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention, which is opposite from Dr.Probe or TFS UI display
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 13,
    'N_scans'           : 50688,
    'N_scan_slow'       : 256,
    'N_scan_fast'       : 198,
    'scan_step_size'    : 0.415, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'obj_tilts'         : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # (tilt_y,tilt_x)mrad, 'tilt_type' = 'all' or 'each'
    'omode_max'         : 1, #1
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6, #2
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,1,3,2),
    'cbeds_reshape'     : (50688,128,128),
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': [exp_CBED_path, 'cbed'],
    'obj_source'         : 'simu',
    'obj_params'         : None, # (1,8,391,403),
    'probe_source'       : 'simu',
    'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None,
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}