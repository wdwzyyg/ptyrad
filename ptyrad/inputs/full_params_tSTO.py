# tSTO

ptycho_output_path = 'output/tSTO/20240604_full_N65536_dp128_random32_p6_1obj_15slice_dz10.0_plr5e-4_oalr5e-4_oplr5e-4_slr5e-4_tlr0_orblur0.4_ozblur1_oathr0.95_opos_dpblur0.5_sng1.0_spr0.1/model_iter0100.pt'
exp_CBED_path      = 'data/20240412_STO_bilayer_07_Hari/data_roi1_Ndp256_dp.hdf5' 

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
    'N_scans'           : 65536,
    'N_scan_slow'       : 256,
    'N_scan_fast'       : 256,
    'scan_step_size'    : 0.393, # Ang
    'scan_flipT'        : (0,0,1), # (0,0,1) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : (0.96,0.14,0,6), #(1,0,2,0), # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1,
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 6,
    'pmode_init_pows'   : [0.02],
    'probe_permute'     : None,
    'cbeds_permute'     : (0,2,1),
    'cbeds_reshape'     : None,
    'cbeds_flipT'       : None, # Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'probe_simu_params' : None
    }

# Source and params, note that these should be changed in accordance with each other
source_params = {
    'measurements_source': 'hdf5',
    'measurements_params': {'path': exp_CBED_path, 'key':'dp'},
    # 'obj_source'         : 'simu', 
    # 'obj_params'         : None,
    # 'probe_source'       : 'simu',
    # 'probe_params'       : None, 
    'pos_source'         : 'simu',
    'pos_params'         : None,
    'tilt_source'        : 'simu',
    'tilt_params'        : {'tilt_type':'all', 'init_tilts':[[0,0]]}, # 'init_tilts' = (tilt_y,tilt_x) mrad, 'tilt_type' = 'all', 'each' 
    'obj_source'         : 'PtyRAD', 
    'obj_params'         : ptycho_output_path, 
    'probe_source'       : 'PtyRAD',
    'probe_params'       : ptycho_output_path, 
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
    'recenter_cbeds'      : None,    # 'all', 'each', None
    'detector_blur_std'   : 0.5,    # scalar(px), None
    'lr_params':{
        'obja'            : 5e-4,
        'objp'            : 5e-4,
        'obj_tilts'       : 0, 
        'probe'           : 5e-4, 
        'probe_pos_shifts': 5e-4}}

loss_params = {
    'loss_single': {'state': True, 'weight': 1.0, 'dp_pow': 0.5},
    'loss_poissn': {'state': False, 'weight': 1.0, 'dp_pow':1.0},
    'loss_pacbed': {'state': False, 'weight': 0.5, 'dp_pow': 0.2}, # Usually weight:0.5, dp_pow:0.2
    'loss_sparse': {'state': True, 'weight': 0.1, 'ln_order': 1},
    'loss_simlar': {'state': False, 'weight': 0.2, 'obj_type':'both', 'scale_factor':[1,1,1], 'blur_std':1}
}

constraint_params = {
    'ortho_pmode'   : {'freq': 1},
    'probe_mask_k'  : {'freq': None, 'radius':0.22, 'width':0.05}, # k-radius should be larger than 2*rbf/Npix to avoid cutting out the BF disk
    'fix_probe_int' : {'freq': 1},
    'obj_rblur'     : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':0.4}, # Ideally kernel size is odd and larger than 6std+1 so it decays to 0
    'obj_zblur'     : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':1},
    'kr_filter'     : {'freq': None,    'obj_type':'both', 'radius':0.15, 'width':0.05},
    'kz_filter'     : {'freq': None,    'obj_type':'both', 'beta':0.2, 'alpha':1},
    'obja_thresh'   : {'freq': 1, 'relax':0, 'thresh':[0.5**(1/15), 1.5**(1/15)]},
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
output_dir   = 'output/tSTO'
prefix       = get_date(date_format='%Y%m%d')
postfix      = 're1a_aff0.96_0.14_0_6'
fig_list     = ['loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos'] # 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all' for all the figures