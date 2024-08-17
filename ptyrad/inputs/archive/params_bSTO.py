# bSTO

#ptycho_output_path = 'data/20240508_bSTO_Hari/Niter100.mat'
ptycho_output_path = 'output/bSTO/20240602_full_N50688_dp128_random32_p6_1obj_13slice_dz10.0_plr1e-4_oalr5e-4_oplr5e-4_slr1e-4_tlr1e-4_orblur0.4_ozblur1_opos_dpblur0.5_sng1.0_spr0.1_ozk5_oathr0.8_pretilt_re1a/model_iter0200.pt'
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
    'scan_flipT'        : None, # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1, #1
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6, #2
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'meas_permute'           : (0,1,3,2),
    'meas_reshape'           : (50688,128,128),
    'meas_flipT'             : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'meas_crop'              : None, # None or (4,2) array-like for [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]
    'meas_resample'          : None, # None or (2,1) array-like for [ky_zoom, kx_zoom]
    'meas_add_source_size'   : None, # None or a scalar of std (Ang)
    'meas_add_detector_blur' : None, # None or a scalar of std (px)
    'meas_add_poisson_noise' : None, # None or a scalar of electrons/Ang^2
    'probe_simu_params'      : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'mat',
    'measurements_params': {'path': exp_CBED_path, 'key':'cbed'},
    # 'obj_source'         : 'simu',
    # 'obj_params'         : None,
    # 'probe_source'       : 'simu',
    # 'probe_params'       : None, 
    # 'pos_source'         : 'simu',
    # 'pos_params'         : None,
    'tilt_source'        : 'simu',
    'tilt_params'        : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # 'init_tilts' = (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each' 
    'obj_source'         : 'PtyRAD', 
    'obj_params'         : ptycho_output_path, 
    'probe_source'       : 'PtyRAD',
    'probe_params'       : ptycho_output_path, 
    'pos_source'         : 'PtyRAD',
    'pos_params'         : ptycho_output_path,
    # 'tilt_source'        : 'PtyRAD',
    # 'tilt_params'        : 'ptycho_output_path,
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}