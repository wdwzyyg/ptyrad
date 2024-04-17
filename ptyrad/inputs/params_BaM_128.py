# BaM_128

ptycho_output_path = 'data/20240412_BaM_2022_03_25_data07_255pix_44L_reg0p5_padded_Hari/Niter1000_128_bilinear.mat'
exp_CBED_path      = 'data/20240412_BaM_2022_03_25_data07_255pix_44L_reg0p5_padded_Hari/data_roi1_Ndp128_dp.hdf5' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 30.620, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.0904*2,# Ang
    'defocus'           : -102.09, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 5, # Ang
    'Nlayer'            : 44,
    'N_scans'           : 65025,
    'N_scan_slow'       : 255,
    'N_scan_fast'       : 255,
    'scan_step_size'    : 0.415, # Ang
    'scan_flip'         : None, 
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'omode_max'         : 1,
    'pmode_max'         : 6,
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
    'obj_params'         : None, #ptycho_output_path, #(1,44,860,860)
    'probe_source'       : 'simu',
    'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None, 
    'omode_occu_source'  : 'uniform',
    'omode_occu_params'  : None
}
