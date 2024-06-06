# NMC

#ptycho_output_path = 'data/20240508_NMC_Dasol/4/roi0_Ndp128/MLs_L1_p4_g32_Ndp128_pc1_noModel_vp1_Ns10_dz10_reg0.7/Niter1000.mat'
ptycho_output_path = 'output/NMC/20240512_full_N4096_dp128_sparse64_p4_plr1e-4_oalr5e-4_oplr5e-4_slr1e-4_tlr5e-4_1obj_10slice_dz10.0_kzreg1_pmk0.52/model_iter2000.pt'
exp_CBED_path      = 'data/20240508_NMC_Dasol/4/data_roi0_Ndp128_dp.hdf5' 

exp_params = {
    'kv'                : 300,  # kV
    'conv_angle'        : 18.3, # mrad, semi-convergence angle
    'Npix'              : 128, # Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
    'rbf'               : None, # Pixels of radius of BF disk
    'dx_spec'           : 0.2690,# Ang, used to calculate dk
    'defocus'           : -100, # Ang, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention, which is opposite from Dr.Probe or TFS UI display
    'c3'                : 0, # Ang, spherical aberration coefficients
    'z_distance'        : 10, # Ang
    'Nlayer'            : 10,
    'N_scans'           : 4096,
    'N_scan_slow'       : 64,
    'N_scan_fast'       : 64,
    'scan_step_size'    : 0.246, # Ang
    'scan_flipT'        : (0,1,0), # (0,1,0) for 'simu' pos, None for loaded pos. Modify scan_flipT would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
    'scan_affine'       : None, #(1,0,-2.5,-2), # (scale, asymmetry, rotation, shear)
    'scan_rand_std'     : 0.15, # None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
    'omode_max'         : 1, #1
    'omode_init_occu'   : {'occu_type':'uniform', 'init_occu':None},
    'pmode_max'         : 4, #2
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
    # 'pos_source'         : 'simu',
    # 'pos_params'         : None,
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