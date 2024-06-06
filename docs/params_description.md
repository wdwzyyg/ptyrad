# Params description for PtyRAD
Chia-Hao Lee

cl2696@cornell.edu

Created 2024.05.24

## Intro 
This .md doc gives some mild introduction to the available parameters/options of PtyRAD. See `ptyrad.inputs.full_params_tBL_WSe2.py` for an actual example.

## file paths
- `ptycho_output_path`: Path to your previous ptycho reconstructions, currently supporting `PtychoShelves` and `PtyRAD`
- `exp_CBED_path`     : Path to your experimental 4D-STEM data, currently supporting `.mat`, `.tif`, `.hdf5`

## exp_params
- `kv`                : Acceleration voltage of the electron in kV
- `conv_angle`        : semi-convergence angle in mrad
- `Npix`              : Detector pixel number, EMPAD is 128. Only supports square detector for simplicity
- `rbf`               : Pixels of radius of BF disk, usually set it to `None` so the calibration is done by `dx_spec`
- `dx_spec`           : Real space pixel size in Ang, used to calculate dk
- `defocus`           : Defocus in Ang for simulated initial probe, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland/abtem/ptychoshelves convention, which is opposite from Dr.Probe or TFS UI display
- `c3`                : Spherical aberration coefficient in Ang for simulated initial probe
- `z_distance`        : Slice thickness in Ang for multislice ptychography
- `Nlayer`            : Number of z-slices in the object
- `N_scans`           : Total number of scan positions (or diffraction patterns)
- `N_scan_slow`       : Number of scan positions along the slow scan direction, usually it's the vertical direction of acquisition GUI
- `N_scan_fast`       : Number of scan positions along the fast scan direction, usually it's the horizontal direction of acquisition GUI
- `scan_step_size`    : Step size between scan positions in Ang
- `scan_flipT`        : Additional flip and transpose for scan patterns. Usually set to `None` for loaded pos. Note that modifying `scan_flipT` would change the image orientation. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
- `scan_affine`       : Global affine transformation for the scan pattern. (scale, asymmetry, rotation, shear)
- `scan_rand_std`     : None or scalar. Randomize the initial guess of scan position with Gaussian distributed displacement (std in px) to avoid raster grid pathology
- `omode_max`         : Maximum number of mixed object modes. For simulated initial object, it'll be generated with the same number of object modes. For loaded object, the omode dimension would be capped at this number
- `omode_init_occu`   : Specified weight (occupancy) for each object weights. Typically we do 'uniform' for frozen phonon like configurations. {'occu_type':'uniform', 'init_occu':None},
- `pmode_max`         : Maximum number of mixed probe modes. For simulated initial probe, it'll be generated with the same number of probe modes. For loaded probe, the pmode dimension would be capped at this number
- `pmode_init_pows`   : Initial power for each additional probe modes. If set at [0.02], that means all additional probe modes would contain 2% of the total intensity
- `probe_permute`     : Additional permutation of the initialized probe. This is usually set as `None` 
- `cbeds_permute`     : Additional permutation of the initialized CBED. This is usually needed if you're loading a 4D CBED file
- `cbeds_reshape`     : Additioanl reshaping of the initialized CBED. This is usually needed if you're loading a 4D CBED file(N_scans,ky,kx)
- `cbeds_flipT`       : Additional flip and transpose for the initialized CBED. Expected input is [flipup, fliplr, transpose] just like PtychoShleves
- `probe_simu_params` : Additional probe simulation parameters (aberration coefficients)

## source_params
- `measurements_source`: Data type of the CBED patterns. Currently supporting `.mat`, `.tif`, `.hdf5`
- `measurements_params`: Path and dict key to retrieve the data matrix. For example: [exp_CBED_path, 'cbed']. For py4dstem processed cbeds, use `'/datacube_root/datacube/data'` to retrieve the 4D-STEM data
- `obj_source`         : Source of the object. Currently supporting `simu`, `custom`, `PtyShv`, and `PtyRAD`
- `obj_params`         : Params of the object. Provide `ptycho_output_path` if you're loading from `PtyShv` or `PtyRAD`, provide `None` for `simu`, and provide np array for `custom`
- `probe_source`       : Source of the probe. Currently supporting `simu`, `custom`, `PtyShv`, and `PtyRAD`
- `probe_params`       : Params of the probe. Provide `ptycho_output_path` if you're loading from `PtyShv` or `PtyRAD`, provide `None` or `probe_simu_params` for `simu`, and provide numpy array for `custom`
- `pos_source`         : Source of the position. Currently supporting `simu`, `custom`, `PtyShv`, and `PtyRAD`
- `pos_params`         : Params of the position. Provide `ptycho_output_path` if you're loading from `PtyShv` or `PtyRAD`, provide `None` for `simu`, and provide np array for `custom`
- `tilt_source`        : Source of the object tilt. Currently supporting `simu`, `custom`, and `PtyRAD`
- `tilt_params`        : Params of the object tilt. Provide `path` if you're loading fro `PtyRAD`, provede a dict like `{'tilt_type':'each', 'init_tilts':[[0,0]]}` if you're using `simu`. `'tilt_type'` can be `'each'` or `'all'`. Provide `'init_tilts'` as an (1,2) array (tilt_y,tilt_x) mrad

Note that the obj, probe, and pos do not need to be coming from the same type of source. You can mix-match PtychoShelves results with PtyRAD results, or load probes/obj/pos from different paths as long as you have the correct pair of [source, params]

## model_params
- `recenter_cbeds`      : Shift experimental CBEDs to reduce linear phase ramp in the object. Usually set to None, but `all` (shift all CBEDs by the CoM of the sum of CBED), and `each` (shift all CBEDs by their own CoM) are available. Not recommended if your CBED is noisy and contains sparse px intensities
- `detector_blur_std`   : Detector Gaussian blur in detector px (dk), usually used for simulating the PSF of high-energy electron on detectors
- `lr_params`: Learning rate dict for optimizable variables (tensors). Usually slower learning rate leads to better convergence/results, but is also updating slower. Only the variable with non-zero learning rate would be optimized. If you don't want/need to optimize certain parameters, set their learning rate to 0 for faster computation. Typical learning rate is 1e-3 to 1e-4.
    ```
    'obja'            : 5e-4, # Learning rate for object amplitude
    'objp'            : 5e-4, # Learning rate for object phase
    'obj_tilts'       : 0,    # Learning rate for object tilts
    'probe'           : 1e-4, # Learning rate for probe
    'probe_pos_shifts': 1e-4, # Learning rate for probe position
    ```        

## loss_params
- `loss_params`: Use `state` to switch on/off each loss term, and use `weight` to determine their relative importance. Each loss term would generate their corresponding grad to the variable, and the final update is determined by the sum of all grad coming from all participating loss terms. Some loss terms have finer control params like `dp_pow` would raise the diffraction pattern to a power before the calculation, and `ln_order` means the L_n norm used to regularize object phase (`objp`) 
    ```
    'loss_single': {'state': True, 'weight': 1.0, 'dp_pow': 0.5}, 
    # MSE loss for a single CBED, same as the maximum-likelihood with Gaussian statistics (high dose)

    'loss_poissn': {'state': True, 'weight': 1.0, 'dp_pow':1.0, 'eps':0},
    # Poisson loss for a single CBED, same as the maximum-likelihood with Poisson statistics (low dose)

    'loss_pacbed': {'state': True, 'weight': 0.5, 'dp_pow': 0.2}, 
    # Loss that compares the PACBED, should use it with compact grouping and a reduced `dp_pow` to emphasize the DF region of the CBED

    'loss_sparse': {'state': True, 'weight': 0.1, 'ln_order': 1}, 
    # Sparsity constrain that penalize any non-zero values. This is applied on the object phase (`objp`) to promote cleaner background and essentially pushes the value towards 0 if you set `ln_order=1`. The `ln_order` stands for the L_n norm. L1 loss is essentially Loss = sum(|x|)

    'loss_simlar': {'state': False, 'weight': 1.0, 'obj_type':'both', 'scale_factor':[1,1,1], 'blur_std':1}, 
    # Similarity loss designed for regularizing object modes by enforcing their similarity so that we don't have missing/extra atoms between the modes. The similarity is enforced by minimizin the standard deviation along the omode dimension with scaled/blurred object.
    ```

## constraint_params
- `constraint_params`: Use `freq` to control how often you want to apply these iteration-wise constraints. `freq=1` means executing after every iteration. Set `freq=None` if you don't want that constraint. Unlike the `lr_params` and `loss_params`, these constraint operations have very little impact on the computation time because they're done in iteration-wise.
    ```
    'ortho_pmode'   : {'freq': 1}, 
    # Apply the same SVD decomposition and orthogonalization of the mixed-state probe as PtychoShelves (except this implementation is vectorized). You should keep this on if you're doing mixed-state probe.
    
    'probe_mask_k'  : {'freq': None, 'radius':0.22, 'width':0.05}, 
    # Apply a k-space sigmoid (similar to a top-hat) probe mask that prevents the probe from absorbing the object structure in k-space. It might cause high-frequency oscillations in the real-space object if you have strong diffuse background in the CBED and did not provide mixed-object to properly reproduce it. Recommend setting it to `None` unless you're pursuing mixed-object with more physical probes .
    
    'fix_probe_int' : {'freq': 1}, 
    # Apply a scaling to probe intensity to make it consistent with the total CBED intensity, essentially fixing the probe intensity. This is needed to stabilize the object amplitude update because the probe update could change the total intensity. This removes the scaling constant ambiguity between probe and object.
    
    'obj_rblur'      : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':1},
    # Apply a "lateral" 2D Gaussian blur to the object. You may choose whether you want to apply to object amplitude ('obja'), phase ('objp'), or 'both' with a specified 'std' in obj px
    
    'obj_zblur'      : {'freq': 1, 'obj_type':'both', 'kernel_size': 5, 'std':0.4},
    # Apply a "z-direction" 1D Gaussian blur to the object. You may choose whether you want to apply to object amplitude ('obja'), phase ('objp'), or 'both' with a specified 'std' and 'kernel_size' in obj px

    'kr_filter'     : {'freq': None,    'obj_type':'both', 'radius':0.15, 'width':0.05},
    # Apply a "lateral" Fourier low-pass filtering to the object. This is similar to the band-pass filter in Digital Micrograph that the k-space filter has a sigmoid-like profile, essentially a cut-off spatial frequency. Typically we're reconstucting object all the way to kMax (Nyquist frequency) so there's not much room for us to filter out hence it's recommended to keep this off unless you want to exclude certain spatial frequencies.

    'kz_filter'     : {'freq': 1,    'obj_type':'both', 'beta':1, 'alpha':1},
    # Apply the arctan kz filter just like the `multilayer_regularization` in PtychoShelves. This is critical for multislice ptychography and you should keep this on if you're dealing with multislice ptycho.

    'obja_thresh'   : {'freq': None, 'relax':0, 'thresh':[0.95**(1/25), 1.05**(1/25)]},
    # Apply a thresholding of object amplitude around 1. The threshold is defined by 'thresh' and the value is determined by the min and max and then taking the power because object amplitude has multiplicative to the propagating wave amplitude. The thresholding can be relaxed by the `relax` param that is a weighted sum betwen the pre-threshold and post-threshold values.

    'objp_postiv'   : {'freq': 1,    'relax':0},
    # Apply a positivity constrain of the object phase, make it non-negative. This clips the negative values so the object is visually darker but with stronger constrast, it's suggested to keep it on so that you can interpret, compare, and process your object phase with a simple baseline.

    'tilt_smooth'   : {'freq': None, 'std':2}
    # Apply a lateral Gaussian blur of the local object tilts in unit of "scan positions". This smoothens the local tilts so that you don't have drastic changes of object tilts between scan positions.
    ```
## Recon params
- `NITER`        : Total number of iterations. 1 iteration means a full pass of all diffraction patterns. Usually 20-50 iterations can get 90% of the work done with a proper learning rate between 1e-3 to 1e-4.
- `INDICES_MODE` : Field-of-view selection for probe positions, each probe position (or each CBED) has a unique index. You may choose 'full' for all probe positions, 'sub' for subsampling every other probe positions (1/4 probe positions in total) with the full FOV, or 'center' for only the center part (1/4 probe positions) with 1/4 FOV area. Typically you can start fro 'sub' to get an quick idea of the entire object, and then switch to 'full' to refine it, there's no dimension mismatch between the INDICES_MODEs.
- `BATCH_SIZE`   : Batch size, or equivalently the "grouping". This is the number of diffraction patterns that will be simultaneously processed and calculated grad for. Usually smaller batch size leads to better final quality, but smaller batch size would also lead to longer computation time per iteration because the GPU isn't as utilized as large batch sizes (due to less GPU parallelism). Generally batch_size of 32 to 128 is used. For extremely large object (or with a lot of object modes), you'll need to reduce batch_size to save GPU memory as well.
- `GROUP_MODE`   : Spatial distribution of the selected probe position within a group, this is similar to the 'MLs' and 'MLc' in PtychoShelves. Usually 'random' is good enough with small batch sizes. 'compact' is believed to provide best performance with better robustness, although it's converging slower. 'sparse' gives the most uniform coverage on the object so converges the fastest, and is also preferred for reconstructions with few scan positions to prevent any locally biased update. However, 'sparse' for 256x256 scan could take more than 10 mins on CPU just to compute the grouping, hence PtychoShelves automatically switches to 'random' for Nscans>1e3. The grouping is fixed during optimization, but the order between each group is shuffled for every iteration.
- `SAVE_ITERS`   : Period of iterations to save results. `SAVE_ITERS = 5` means saving results every 5 iterations.

# Output folder and pre/postfix params
- `output_dir`   : Output directory. The folder will be automatically created if it doesn't exist.
- `prefix`       : Prefix string that usually set as the reconstruciton date.
- `postfix`      : Postfix string that will be appended to the output folder name, useful for comparing different conditions or continuing a reconstruction.
- `fig_list`     : List of desired output figures. You may pass a list with the following members including 'loss', 'forward', 'probe_r_amp', 'probe_k_amp', 'probe_k_phase', 'pos', 'tilt', or 'all' for all the figures.