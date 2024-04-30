# CNS

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
    'scan_flip'         : None, # (2) for 'simu' pos, None for loaded pos. Modify scan_flip would change the image orientation.
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'omode_max'         : 2, #1
    'pmode_max'         : 2, #2
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
    'measurements_params': [exp_CBED_path, 'dp'],
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