# BaM_128

ptycho_output_path = 'data/20240412_STO_bilayer_07_Hari/Niter1000.mat'
exp_CBED_path      = 'data/20240412_STO_bilayer_07_Hari/data_roi1_Ndp256_dp.hdf5' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 30.0, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.1179,# Ang
    'defocus'           : -134, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 5, # Ang
    'Nlayer'            : 30,
    'N_scans'           : 65536,
    'N_scan_slow'       : 256,
    'N_scan_fast'       : 256,
    'scan_step_size'    : 0.393, # Ang
    'scan_flip'         : (2),  # (2) for simu pos, None for imported pos
    'scan_affine'       : None, #(1,0,2,0), # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1,
    'pmode_max'         : 4,
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
    'obj_source'         : 'simu', #'simu',
    'obj_params'         : None, #ptycho_output_path, #ptycho_output_path, #(1,44,860,860)
    'probe_source'       : 'simu',
    'probe_params'       : None, #ptycho_output_path, 
    'pos_source'         : 'simu',
    'pos_params'         : None, 
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}
