# CNS

ptycho_output_path = 'data/CNS_from_Hari/Niter10000.mat'
exp_CBED_path      = 'data/CNS_from_Hari/240327_fov_23p044A_x_24p402A_thickness_9p978A_step0p28_conv30_dfm100_det70_TDS_2configs_xdirection_Co_0p25_Nb_0_S_0.mat' 

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
    'scan_flip'         : (2), # (2) for 'simu' pos, None for loaded pos. Modify scan_flip would change the image orientation.
    'scan_affine'       : (1,0,2,0), # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1, #1
    'pmode_max'         : 2, #2
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,1,3,2),
    'cbeds_reshape'     : (7134,164,164),
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
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}