# tBL_WSe2

ptycho_output_path = 'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/roi1_Ndp128_step128/MLs_L1_p10_g128_pc0_noModel_updW100_mm_dpFlip_ud_T/Niter9000_v7.mat'
exp_CBED_path      = 'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/data_roi1_Ndp128_step128_dp.hdf5'

exp_params = {
    'kv'                : 80,  # kV
    'conv_angle'        : 24.9, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.1494,# Ang
    'defocus'           : 0, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 8, # Ang
    'Nlayer'            : 1,
    'N_scans'           : 16384,
    'N_scan_slow'       : 128,
    'N_scan_fast'       : 128,
    'scan_step_size'    : 0.4290, # Ang
    'scan_flip'         : None, 
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1,
    'pmode_max'         : 10,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : None,
    'cbeds_reshape'     : None,
    'cbeds_flip'        : (1),
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': [exp_CBED_path, 'dp'],
    'obj_source'         : 'simu', #'PtyShv', 
    'obj_params'         : None, #ptycho_output_path, #(1,1,592,592),
    'probe_source'       : 'simu',
    'probe_params'       : None,  
    'pos_source'         : 'simu',
    'pos_params'         : None, 
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}