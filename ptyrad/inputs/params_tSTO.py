# tSTO

ptycho_output_path = 'output/tSTO/20240604_full_N16384_dp128_random32_p6_1obj_15slice_dz10.0_plr1e-3_oalr1e-3_oplr1e-3_slr5e-4_tlr0_orblur0.4_ozblur1_oathr0.99_opos_dpblur0.5_sng1.0_spr0.1/model_iter0050.pt'
exp_CBED_path      = 'data/20240412_STO_bilayer_07_Hari/data_roi1_Ndp128_dp.mat' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 30.0, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.1179,# Ang
    'defocus'           : -134, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 15,
    'N_scans'           : 16384,
    'N_scan_slow'       : 128,
    'N_scan_fast'       : 128,
    'scan_step_size'    : 0.393, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : (0.96,0.14,0,6), #(1,0,2,0), # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : None, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (2,0,1), # (0,2,1) for Ndp256_dp.hdf5
    'cbeds_reshape'     : None,
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': {'path': exp_CBED_path, 'key':'cbed'},
    # 'obj_source'         : 'simu', 
    # 'obj_params'         : None,
    # 'probe_source'       : 'simu',
    # 'probe_params'       : None, 
    'tilt_source'        : 'simu',
    'tilt_params'        : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # 'init_tilts' = (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each' 
    # 'obj_source'         : 'PtyRAD', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyRAD',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyRAD',
    # 'pos_params'         : ptycho_output_path,
    # 'tilt_source'        : 'PtyRAD',
    # 'tilt_params'        : 'ptycho_output_path,
    # 'pos_source'         : 'PtyRAD',
    # 'pos_params'         : ptycho_output_path,
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}