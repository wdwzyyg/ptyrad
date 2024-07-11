# bSTO

ptycho_output_path = 'output/bSTO/20240602_full_N50688_dp128_random32_p6_1obj_13slice_dz10.0_plr0_oalr5e-4_oplr5e-4_slr0_tlr0_orblur0.4_ozblur1_opos_dpblur0.5_sng1.0_spr0.1_ozk5_oathr0.8_pretilt/model_iter0460.pt'
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
    'scan_rand_std'     : None, # 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
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

model_params = {
    'obj_preblur_std'     : None, # scalar(px), None
    'detector_blur_std'   : 0.5,  # scalar(px), None
    'lr_params':{
        'obja'            : 5e-4,
        'objp'            : 5e-4,
        'obj_tilts'       : 1e-4, 
        'probe'           : 1e-4, 
        'probe_pos_shifts': 1e-4}}

loss_params = {
    'loss_single': {'state': True, 'weight': 1.0, 'dp_pow': 0.5},
    'loss_poissn': {'state': False, 'weight': 1.0, 'dp_pow':1.0},
    'loss_pacbed': {'state': False, 'weight': 0.5, 'dp_pow': 0.2}, # Usually weight:0.5, dp_pow:0.2
    'loss_sparse': {'state': True, 'weight': 0.5, 'ln_order': 1},
    'loss_simlar': {'state': False, 'weight': 0.2, 'obj_type':'both', 'scale_factor':[1,1,1], 'blur_std':1}
}

constraint_params = {
    'ortho_pmode'   : {'freq': 1},
    'probe_mask_k'  : {'freq': None, 'radius':0.22, 'width':0.05}, # k-radius should be larger than 2*rbf/Npix to avoid cutting out the BF disk
    'fix_probe_int' : {'freq': 1},
    'obj_rblur'     : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':0.4}, # Ideally kernel size is odd and larger than 6std+1 so it decays to 0
    'obj_zblur'     : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':1},
    'kr_filter'     : {'freq': None,    'obj_type':'both', 'radius':0.15, 'width':0.05},
    'kz_filter'     : {'freq': None,    'obj_type':'both', 'beta':1, 'alpha':1},
    'obja_thresh'   : {'freq': 1, 'relax':0, 'thresh':[0.8**(1/13), 1.2**(1/13)]},
    'objp_postiv'   : {'freq': 1,    'relax':0},
    'tilt_smooth'   : {'freq': None, 'std':2}
}

def get_date(date_format = '%Y%m%d'):
    from datetime import date
    date_format = date_format
    date_str = date.today().strftime(date_format)
    return date_str

# Recon params
NITER        = 500
INDICES_MODE = 'full'   # 'full', 'center', 'sub'
BATCH_SIZE   = 32
GROUP_MODE   = 'random' # 'random', 'sparse', 'compact' # Note that 'sparse' for 256x256 scan could take more than 10 mins on CPU. PtychoShelves automatically switch to 'random' for Nscans>1e3
SAVE_ITERS   = 10        # scalar or None

# Output folder and pre/postfix, note that the needed / and _ are automatically generated
output_dir   = 'output/bSTO'
prefix       = get_date(date_format='%Y%m%d')
postfix      = ''
fig_list     = ['loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt'] # 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all' for all the figures