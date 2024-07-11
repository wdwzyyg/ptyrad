# tBL_WSe2

ptycho_output_path = 'output/tBL-WSe2/20240529_full_N16384_dp128_flipT100_random32_p16_plr1e-4_oalr5e-4_oplr5e-4_slr1e-4_tlr0_1obj_12slice_dz1.0_kzf0.1_prev_probe_pos_sparse/model_iter0200.pt'#'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/roi1_Ndp128_step128/MLs_L1_p10_g128_pc0_noModel_updW100_mm_dpFlip_ud_T/Niter9000_v7.mat'
exp_CBED_path      = 'data/tBL_WSe2/Fig_1h_24.9mrad_Themis/1/data_roi1_Ndp128_step128_dp.hdf5'

exp_params = {
    'kv'                : 80,  # kV
    'conv_angle'        : 24.9, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk, used to calculate dk
    'dx_spec'           : 0.1494,# Ang
    'defocus'           : 0, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 12, # Ang
    'Nlayer'            : 1,
    'N_scans'           : 16384,
    'N_scan_slow'       : 128,
    'N_scan_fast'       : 128,
    'scan_step_size'    : 0.4290, # Ang
    'scan_flipT'        : None, # None for both 'simu' and loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 16,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 16,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'meas_permute'           : None,
    'meas_reshape'           : None,
    'meas_flipT'             : [1,0,0], # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'meas_crop'              : [[0,64],[0,64],[1,128],[1,128]], # None or (4,2) array-like for [[scan_slow_start, scan_slow_end], [scan_fast_start, scan_fast_end], [ky_start, ky_end], [kx_start, kx_end]]
    'meas_resample'          : [0.5,0.5], # None or (2,1) array-like for [ky_zoom, kx_zoom]
    'meas_add_source_size'   : 1, # None or a scalar of std (Ang)
    'meas_add_detector_blur' : 1, # None or a scalar of std (px)
    'meas_add_poisson_noise' : 1e5, # None or a scalar of electrons/Ang^2
    'probe_simu_params'      : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': {'path': exp_CBED_path, 'key':'dp'},
    'obj_source'         : 'simu',
    'obj_params'         : None,
    'probe_source'       : 'simu',
    'probe_params'       : None,  
    'pos_source'         : 'simu',
    'pos_params'         : None,
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
    # 'obj_source'         : 'PtyShv', 
    # 'obj_params'         : ptycho_output_path, 
    # 'probe_source'       : 'PtyShv',
    # 'probe_params'       : ptycho_output_path, 
    # 'pos_source'         : 'PtyShv',
    # 'pos_params'         : ptycho_output_path,
}

model_params = {
    'obj_preblur_std'     : None, # scalar(px), None
    'detector_blur_std'   : None, # scalar(px), None
    'lr_params':{
        'obja'            : 5e-4,
        'objp'            : 5e-4,
        'obj_tilts'       : 0, 
        'probe'           : 1e-4, 
        'probe_pos_shifts': 5e-4}}

loss_params = {
    'loss_single': {'state': True, 'weight': 1.0, 'dp_pow': 0.5},
    'loss_poissn': {'state': False, 'weight': 1.0, 'dp_pow':1.0, 'eps':1e-6},
    'loss_pacbed': {'state': False, 'weight': 0.5, 'dp_pow': 0.2}, # Usually weight:0.5, dp_pow:0.2
    'loss_sparse': {'state': True, 'weight': 0.1, 'ln_order': 1},
    'loss_simlar': {'state': True, 'weight': 0.1, 'obj_type':'both', 'scale_factor':[1,1,1], 'blur_std':1}
}

constraint_params = {
    'ortho_pmode'   : {'freq': 1},
    'probe_mask_k'  : {'freq': None, 'radius':0.22, 'width':0.05}, # k-radius should be larger than 2*rbf/Npix to avoid cutting out the BF disk
    'fix_probe_int' : {'freq': 1},
    'obj_rblur'     : {'freq': None, 'obj_type':'both', 'kernel_size': 5, 'std':0.2}, # Ideally kernel size is odd and larger than 6std+1 so it decays to 0
    'obj_zblur'     : {'freq': None, 'obj_type':'both', 'kernel_size': 5, 'std':1},
    'kr_filter'     : {'freq': None,    'obj_type':'both', 'radius':0.15, 'width':0.05},
    'kz_filter'     : {'freq': None,    'obj_type':'both', 'beta':0.1, 'alpha':1},
    'obja_thresh'   : {'freq': None, 'relax':0, 'thresh':[0.6**(1/12), 1.4**(1/12)]},
    'objp_postiv'   : {'freq': 1,    'relax':0},
    'tilt_smooth'   : {'freq': None, 'std':2}
}

def get_date(date_format = '%Y%m%d'):
    from datetime import date
    date_format = date_format
    date_str = date.today().strftime(date_format)
    return date_str

# Recon params
NITER        = 200
INDICES_MODE = 'full'   # 'full', 'center', 'sub'
BATCH_SIZE   = 32
GROUP_MODE   = 'random' # 'random', 'sparse', 'compact' # Note that 'sparse' for 256x256 scan could take more than 10 mins on CPU. PtychoShelves automatically switch to 'random' for Nscans>1e3
SAVE_ITERS   = 10        # scalar or None

# Output folder and pre/postfix, note that the needed / and _ are automatically generated
output_dir   = 'output/tBL-WSe2'
prefix       = get_date(date_format='%Y%m%d')
postfix      = ''
fig_list     = ['loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos'] # 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all' for all the figures