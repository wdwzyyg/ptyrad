# PSO_128

ptycho_output_path = 'data/PSO/MLs_L1_p8_g192_Ndp128_pc50_noModel_vp1_Ns21_dz10_reg1/Niter200.mat'
exp_CBED_path      = 'data/PSO/MLs_L1_p8_g192_Ndp128_pc50_noModel_vp1_Ns21_dz10_reg1/PSO_data_roi0_Ndp128_dp.hdf5'

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 21.4, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.1868,# Ang
    'defocus'           : -200, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 21,
    'N_scans'           : 4096,    
    'N_scan_slow'       : 64,
    'N_scan_fast'       : 64,
    'scan_step_size'    : 0.41, # Ang
    'scan_flip'         : (2),  # (2) for 'simu' pos, None for loaded pos
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1,
    'pmode_max'         : 8,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,2,1),
    'cbeds_reshape'     : None,
    'cbeds_flip'        : None,
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': [exp_CBED_path, 'dp'],
    'obj_source'         : 'simu', #'PtyShv', 
    'obj_params'         : None, #ptycho_output_path, #(1,21,320,320),
    'probe_source'       : 'simu', 
    'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None, 
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}
