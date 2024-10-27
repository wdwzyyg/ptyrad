# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [TODO]
## Initialization
- `initialization.py` can probably be refactored a bit, it's too clunky now. Might consider refactor the process into loading, preprocessing, and initialization. This would also make the `exp_params` much clearer by moving some settings to other dicts.
- prepare the object for multislice (interpolate, pad vacuum) and multiobj (duplicate)
- Let forward model generate diffraction patterns with larger kMax than the actual data, and center-crop the diffraction pattern in the forward process before calculating loss. This would require the object and probe to be initialized at higher real space sampling than the data kMax, but it will allow the forward model to properly scatter outside of the collected region on the detector, hence reduce the edge artifact. This is a better approach than padding the experimental diffraction pattern.
- Optional asynchronous loading of the measurements to further save the memory 
## Probe
- fit aberration to k-space probe. py4DSTEM does it with fit each mode with aberration, although I'm not sure whether that's better or not
- Fix the probe corner intensity artifact. Feel like some intrinsic phase instability of complex probe
- Add an active decoupling between probe and object to avoid probe absorbing too much object structure. Could be a deconvolution in either space. Should look into how PtyShv update the probe closer, and maybe implement an illumination-normalized constraint, or just a full option of conventional analytical grad update for probe 
## Scan position
- Add a scan rotation fitting routine from the curl of gradCoM of CBEDs similar to the py4dstem's `solve_for_center_of_mass_relative_rotation` could be very handy 
- Try [4DSTEM-calibration] (https://github.com/ningustc/4DSTEM-Calibration) for position correction
- Try [iCGD] (https://github.com/ningustc/iCGD) into the position constrain
## Mixed object
- NMF and PCA for object modes? Given frozen phonon configurations, what is a good decomposition method?
- Finish the weighted sum of `omode_occu` in `save_results`
## BO
- Add options to select different optuna optimizers and pruners
- Decouple the BO error from reconstruction loss so we can test different setup
## Recon workflow
- Decouple the reconstruction error with data error so that we can reconstruct with whatever target error, while having an independent data error metric 
- Sequential reconstruction (asize_presolve) is also desired (might write a specific notebook to chain them together)
### Utils and plotting
- Apparently plotting and saving matplotlib figure is incredibly slow, it's taking like 1sec/fig and we'll need some improvements
- Visualize radially accumulated intensity for k-space probe
- Add `get_detector_blur` estimation of detector blur from the tapering of vacuum CBED aperture edge and some fitting. Might be able to suggest better dx calibration if we trust the convergence angle. Can probably combine with `get_rbf` routine
- Add `plot_obj_fft` to `visualization` and maybe to `plot_summary` and `save_reuslts` as well. Some windowed log(S) diffractogram or P+S decomposition could be helpful. (http://www.roberthovden.com/tutorial/2015_fftartifacts.html)
- Add a `plot_obj_tilts_interp` for interpolated version of tilt_x, tilt_y for cleaner visualization could be nice
- Add a routine to check for CBED scaling (rbf/convergence angle) and off centering
- Maybe encapsulate PtyRAD into an executable? How to do that with GPU and PyTorch?
### Code clarity
- Run through Ruff formatter for PEP8 code style
- Add type hints
- Refine doc strings (Google style)
- Use Sphinx and Napolean for API documentation on Read the Docs
- Unified the usage of explicit key or .get for dict
- Unified meshgrid usage, naming, and unit would be nice
## PyTorch performance tuning
- Use DataLoader for measurements. This could be a bit slower than directly loading entire measurements into memory, but it allows dataset larger than memory, and also makes parallel on multiple GPUs possible
- Delete used variables for lower memory footprint
- Use in-place operations on tensors don't require grad

## [unrelease]
### Added
- Add `CustomLogger` to notebooks/scripts to log the Python terminal stdout specifically to a log file under the reconstructed folder. If `SAVE_ITERS = None` and no output directory is generated, the log would be saved to `logs`.
### Changed
- Fix the printing error when using `GROUP_MODE = 'sparse'` by making `sparse_batches` a list of arrays so we can do `batch.tolist()` for cleaner printing of the batch_t
- Change the default update step size for `py4dstem` scripts to 0.1 for numerical stability, otherwise the default 0.5 would easily give NaNs on my tBL-WSe2 dataset. Thanks the input from @sezelt
- Simplify `run_py4dstem.py` and `run_py4dstem_detailed_walkthrough.ipynb` by calling the v0.0.2 version of locally modified py4DSTEM repo 

## [v0.1.0-beta2.9] - 2024-10-17
### Added
- Add `py4DTEM` as a new source for object, probe, and positions in params file with inputs from @dsmagiya
- Refine `run_py4dstem.py`, `slurm_run_py4dstem.sub`, and `run_py4dstem_detailed_walkthrough.ipynb`
- Add `probe_prop` as a new option for `save_result` list. It'll save a 2D montage of (Nz x Ny, pmode x Nx) to show how the probe modes propagate through the object
### Changed
- Modify `load_hdf5` so that it can return the entire dict if `dataset_key = None`
- Add a `with torch.no_grad()` block for the saving/plotting block to locally disable autograd
- Move the `init_accelerator` from `PtyRADSolver` class to `utils` to reduce duplicated printing because DDP is only initialized after initializing `accelerate`
- Change the interpolation mode for on-the-fly resampling (torch.nn.functional.interpolate) from 'area' to 'bilinear' to avoid the edge artifact in reconstructed probe 

## [v0.1.0-beta2.8] - 2024-10-14
### Added
- Add x-ray ptychography capability to PtyRAD with the help from @ameyluktuke, @xyin-anl, and @yijiang1
- Add `simu_xray_ptycho.yml` and demo data for x-ray ptychography
- Add `make_fzp_probe` to `utils.py` to simulate x-ray probe generated by Fresnel zone plates
- Add `foldslice_hdf5` as a new position source for most APS instruments that generate hdf5 position files. The hdf5 files are commonly handled in `fold_slice` using `p.src_positions = 'hdf5_pos';`
### Changed
- `illumination_type` is now a required field in the `exp_params` dict inside the yml param files
- Modify `initialization.py` to be compatible with x-ray ptycho including `init_exp_params`, `init_measurments`, `init_probe`, `init_pos`, `init_obj`, and `init_H`

## [v0.1.0-beta2.7] - 2024-10-05
### Added
- Add on-the-fly measurements resampling into `meas_resample` to reduce GPU VRAM usage with negligible performace impairment
- Add `20241005_effect_of_cpu_cores/` under `docs/`
### Changed
- Update `time_sync` by replacing `time.time()` with `time.perf_counter()` so that it can better measure events that are shorter than 1 ms.
- Let `PtyRADSolver.reconstruct()` create an attribute of reconstructed `PtychoAD` model called `PtyRADSolver.reconstruct_results` so that we can do some further work if needed
- Modify `accelerate` branch so that the HuggingFace `accelerate` package becomes optional and can work on environments without `accelerate` installed. This makes it possible to merge `accelerate` branch into `main` and allow Windows users to at least run `PtyRAD` on a single GPU using the exact same codebase.
- Modify `imshift_batch` so that PtyRAD is compatible with older Python version < 3.11 as well
- Simplify the output message of `set_gpu_device()`
- Merge `accelerate` branch into `main` so that it's easier to maintain. Windows users without NCCL do not need to install `accelerate` package, and they can still use a conda environment without `accelerate` to execute PtyRAD from `main` branch

## [v0.1.0-beta2.6] - 2024-09-30
### Added
- Add `--gpuid` as a command line argument for `run_ptyrad.py` @sezelt
- Add `set_gpu_device` to `utils` for cleaner `run_ptyrad.py` script
- Add `c5` explicitly under `exp_params` so we have more control on simulated initial probe
- Add `c3` and `c5` into hypertunable params @sezelt
- Enable `'step': null` inside `'tune_params'` for continuous parameter space in hypertune mode @sezelt
- Add multi-GPU and mixed-precision capabilities via HuggingFace `accelerate` package. This is implemented in a separate branch `accelerate` and would require a different environment.
### Changed
- Update `get_default_probe_simu_params` to take in 'c5' from `exp_params` as well
- Update `make_output_folder` to include c3, c5 values under the `init` condition inside `recon_dir_affixes`
- Update `make_stem_probe` with more robust param parsing @sezelt
- Update `run_ptyrad` script and notebooks, and `slurm_run_ptyrad.sub` accordingly

## [v0.1.0-beta2.5] - 2024-09-19
### Added
- Add `optimizer_params` dict under `model_params` in the .yml params file to support more PyTorch available optimizers with configurations and allow loading the optimizer state
- Add `create_optimizer` function under `optimization.py` for arbitrary PyTorch optimizer creation with configurations
- Add `grad_accumulation` into `BATCH_SIZE` so we can approximate large batch size that doesn't fit into memory by accumulating gradients from many sub-batches. This is essentially a "memory-save" mode for PtyRAD
### Changed
- Refine the `plot_forward_pass` default indices generation method so it works better for different INDICES_MODE
- Update `make_output_folder` so the optimizer name can be optionally affixed
- Change `make_save_dict` and `save_results` so that it saves the optimizer state into `model.pt` as well
- Rename `model.set_optimizer_params` into `model.create_optimizable_params_dict` for clarity
- Move some jupyter notebooks into `scripts/analysis/` for clarity
- Move `spec-file_ptyrad.txt` to new folder `envs/` for clarity

## [v0.1.0-beta2.4] - 2024-09-17
### Added
- Add an `'obja'` option to the `save_results` to allow saving the object amplitude. 
### Changed
- Change the `plot_forward_pass` default behavior in `plot_summary` from random indices to fixed indices so the reconstruction progress can be better observed by visualizing the same region throughout the run.
- Fix normalization error of `'bit: ['raw']` in `'result_modes'` since beta2.2 (2024-09-03). It was incorrectly normalizing the tif outputs from 0 to 1 when it should be outputting the original range. The output figure like `forward` and saved optimized tensors in `model.pt` were not affected by this error.

## [v0.1.0-beta2.3] - 2024-09-13
### Added
- Add a simple notebook `check_sqlite.ipynb` to check duplicated params in sqlite database for hypertune mode, though the duplicatation is an expected behavior for BO algorithm
- Add `raw` as new measurement data source to handle EMPAD and pre-processed EMPAD2 4D-STEM datasets
- Add `power_thresh` to `probe_mask_k` constraint so we can select how much probe modes should be masked in k-space
### Changed
- Specify the file_path in all loading functions in `data_io` when there's a `FileNotFoundError`
- Add `indices` into the argument of `make_save_dict` so that the selected probe position indices are saved into `model.pt` as well. This enables more convenient custom object cropping.
- `lr_params` is merged with `start_iter` and renamed to `update_params` under `model_params` to add extra control over when to start optimizing the optimizable tensors
- modify `make_output_dir` to affix non-zero `start_iter`, also change the `lr` affix to non-zero learning rates only

## [v0.1.0-beta2.2] - 2024-09-03
### Added
- Add a default `/data` folder with txt instruction
- Add `demo` under `params` with a couple tBL_WSe2 examples
- Add `run_PtyShv.m` and `slurm_run_PtyShv.sub` for direct comparison with PtychoShelves
- Add `copy_params` boolean to `recon_params` to copy the params files to the output directories for better record keeping
- Add `save_results` list to `recon_params` to specify which result (obj, probe) to save
- Add `result_modes` dict to `recon_params` to specify the dimension of output object and whether to postprocess (crop, bit depth) the result before saving
- Add `collate_results` boolean to `hypertune_params` to specify whether to collect hypertune results under `output_dir`
- Add `parse_sec_to_time_str` to `utils` to display the solver time and iteration time in flexible time string from days, hours, mins, to secs
- Add full description to every entry in the params file
### Changed
- Move `subscan_slow` and `subscan_fast` under `INDICES_MODE` for (hopefully) clarity
- Let `load_params` add additional entry of `params_path` to the params dict before return for easier usage of `copy_params`
- Absorb `--hypertune` and `--quiet` into the params file, simplifying the scripts and letting the entire recontstruction behavior controlled by params file
- Drop the `_optuna` suffix in `README.md` and `spec-file.txt` for simplicity
- Simplify the installation guide in `README.md`
- Rename `fig_list` to `selected_figs` for clarity
- Rename `dir_affixes` to `recon_dir_affixes` for clarity
### Removed
- Remove `probe_simu_params` from `exp_params` because it's duplicated with `probe_params` in `source_params`

## [v0.1.0-beta2.1] - 2024-08-28
### Added
- Add `get_scan_affine.ipynb` to quickly estimate the scan affine transformation for known crystal structure
- Add `decompose_affine_matrix` to `utils` to decompose an affine matrix into the 4 components
- Add `subscan_slow` and `subscan_fast` into `recon_params` for finer control of `INDICES_MODE` like `center` and `sub`
- Add `dir_affixes` to `recon_params` to enable flexible control of the output folder name with `make_output_folder`
- Add `defocus` and `conv_angle` to Optuna optimizable params in `hypertune_params` 
### Changed
- Fix `optuna_objective` so that the 4 components of `scan_affine` can be optimized independently
- Move `inputs` out of `ptyrad` core package and rename it as `params` for simplicity
- Rename the `full_params_xxx.yml` into `xxx.yml` for simplicity
- Modify `LoopSubmit.sh` so the 1st loop would wait 10 sec before the 2nd one to finish the database creation, which prevents the sqlite3 "table already exists error"
- Fix `make_mixed_probe` arguments with optional verbose

## [v0.1.0-beta2.0] - 2024-08-18
### Added
- Add hyperparameter tuning (Bayesian optimization and others) capability for `z_distance`, `scan_affine`, and `obj_tilts` with Optuna
- Add `reconstruction` module and `PtyRADSolver` wrapper class for a more compact workflow and streamlined interface
- Add `load_params` into `data_io`
- Add `.yml` as a new params file type. The original `.py` is still working but deprecated, might be removed before public release
- Add `vprint` as verbose print to `utils` to better control the verbosity of printed information (especially for hyperparamter tuning)
- Add a rough version of doc string for major classes and functions
### Changed
- Simplify the arguments for `save_results` and `make_save_dict`
- Rearrange the argument order of `plot_summary` and add default value to `fig_list`
- Refactor the `run_ptyrad` scripts and notebooks and move them into `scripts` 
- Move previous params.py files to `ptyrad/inputs/archive` and stop updating them
- Fix `make_output_folder` so that dz will only print 3 significant figures after rounded to 2 decimal points. So dz = 12.8 would be printed as 12.8 instead of 12.800000190734863.
- Add the `eps=1e-10` back to `multislice_forward_model_vec_all` in `forward` so that the `dp.pow()` is more numerically stable, especially for large collection angles with intensities near 0
### Removed
- Remove `rbf` from `exp_params` and `make_stem_probe`. This is to encourage/enforce users to calibrated their dk for each camera length to get `dx_spec`.
- Remove `make_recon_params_dict` from `utils`
- Remove `load_empad_as_4d` and `save_4D_as_hdf5` and other archived from `data_io` because we're not using it at the moment

## [v0.1.0-beta1.3] - 2024-07-10
### Added
- Add 4D-STEM preprocessing methods including `meas_crop`, `meas_resample`, `meas_add_source_size`, `meas_add_detector_blur`, `meas_add_poisson_noise` to `initialization` for better handling of input simulated 4D-STEM data. With these new methods, users can easily reconstruct with different 4D-STEM data conditions without manually generating and saving each 4D-STEM variants. We may create 4D-STEM datasets with different collection angles, k-space sampling, partial spatial coherence, detector blur, and noise level from a single dataset right before the reconstruction.
- Add `obj_preblur_std` to `model` for an effective real space deconvolution with a 2D Gaussian kernel. By pre-convolving the obj with a 2D Gaussian before simulating the diffraction pattern, the reconstructed obj is essentially the deconvolution version of the transmission function.
### Changed
- Rename `cbeds` variables/keys into `meas` or `DP` for generalizability. Changes are primarily made inside `initialization`, `optimization`, and `visualization` but you will need to modify the params files.
- Modify `make_output_folder` to include `obj_preblur_std` values
- Modify the gaussian_blur implementation in `loss_simlar` from a stack/list comp version to a reshape version and gets a 25% speed up (45 sec vs. 1min /iter)!
### Removed
- Remove `recenter_cbeds` in `model_params` because sub-px shifting noisy CBEDs is really not a good idea and leaves quite some artifact as well. We should directly handle the obj linear phase ramp from the off-centered CBEDs.
- Remove `run_ptyrad_local.py` for simplicity as I don't expect a lot of users would need it anymore. One can modify the atlas script by changing the path of `ptyrad` package.

## [v0.1.0-beta1.2] - 2024-06-05
### Added
- Add `obj_zblur` to `optimization` for a real-space substitution of `kz_filter`. By convolving a 1D Gaussian filter along z-direciton, we could remove the wrap-around while maintaining the z-regularization behavior. Note that there's no free lunch so instead of the wrap-around from `kz_filter`, the `obj_zblur` would still introduce edge effect due to the convolution. The default is "same" padding with "replicate" padding mode, so the object is padded with edge elements like abc|ccc, where | stands for the object edge 
- Add `get_decomposed_affine_matrix` to `utils` to quickly estimate the needed scan affine transformation components if we already have a reconstructed object and we know the ideal lattice constant and the angle between lattice vectors
### Changed
- Move `obj_tilts` to `source_params` so that we can decouple it with the `init_cache` and use it freely from scratch (e.g. start from random object, probe, pos but with known local tilts from previous reconstructions)
- Change the `measurements_params` for `'mat'` and `'hdf5'` from a `list` to a `dict` for better clarity, i.e. [path, 'cbed'] -> {'path':<path>, 'key':'cbed'}
- Modify `make_output_folder` to include `obja_thresh` values in accordance with the added `obj_zblur` feature. Because the `kz_filter` automatically contains a soft thresholding for obja so in order to fully replace it with `obj_zblur`, we'll need to additionally specify `obja_thresh` as well
- Decouple the obj tilts from `plot_scan_pos` by adding `plot_obj_tilts` to `visualization`

## [v0.1.0-beta1.1] - 2024-05-31
### Added
- Add `scan_rand_std` option to `initialization` for Guassian displacements of scan positions to reduce the raster grid pathology
- Add `loss_poissn` to `optimization` for loss calculation with Poisson noise statistics. This should be helpful for low dose data.
### Changed
- Add the `show_fig` flag and `plt.ioff` to `plot_pos_grouping` so that it's consistent with other plotting functions
- Remove the `os.environ["OMP_NUM_THREADS"] = "4"` in  `utils` since I somehow don't get the warning from `MiniBatchKMeans` anymore
- Add the description about reading py4dstem-processed .hdf5 with data key `'/datacube_root/datacube/data'` found by Desheng to `params_description.md`
- Rearrange output strings in `make_output_folder` and add keywords (`show_lr`, `show_constraint`, `show_model`, `show_loss`)to disable optional info like learning rates, constraints, detector blur, and losses

## [v0.1.0-beta1.0] - 2024-05-23
### Added
- Add `blur_std` option for `loss_simlar` so we could compare the std between Gaussian blurred object modes
- Add `kr_filter` as additional obj constraint to `optimization` so we could set our desirable maximum kr that contributes to the object image. Note that the positivity clipping could be doubling the apparent spatial frequencies
- Add `run_ptyrad_altas.py`, `run_ptyrad_local.py`, and `slurm_run_ptyrad.sub` as demo scripts
- Add `docs/` and `params_description.md` for some more explanation
### Changed
- Rename the `kz_filter` output folder string in `make_output_folder` from `kzreg` into `kzf` for simplicity, and to be consistent with `krf` for `kr_filter` 
- Change the `shift_cbeds` in `models` from `.abs()` to `.real.clamp(min=0)` because it seems that taking the real part of the complex Fourier filtered output from a real-valued input is a more correct approach
- Use the `plt.show(block=False)` flag for `plot_pos_grouping` so that the non-interactive python execution wouldn't be block by the `plt.show()` when executed from script
- Update `README.md` with the new conda install commands for local and altas usage

## [v0.1.0-alpha3.10] - 2024-05-20
### Added
- Add `load_PtyRAD` option for `obj_tilts` initialization in `exp_params` so we can reconstuct from previous PtyRAD runs with `opt_obj_tilts`
- Add `get_local_obj_tilts` estimation function into `utils`
- Add `get_local_obj_tilts.ipynb` notebooks to `scripts/` for interactive demonstration
- Update `CHANGELOG.md` and fill in previous record information for record keeping
### Changed
- Rename `source_params` keyword in `initialization` from `pt` to `PtyRAD` for clarity
- Change the comments in params files to hint the expected keywords for `obj_tilts`
### Removed
- Remove `z_pad` argument from `kz_filter` because it's not as helpful. The obj still shows artificial intensity decay for top and bottom layers for z_pad = 2 and 8 and can not fix the multislice obj wrap-around artifact
- Remove `pphase_smooth` constraint because it's not as helpful. Naively modifying (smoothing or scaling) the Fourier probe phase has a rough effect of focusing the real space probe and doesn't really stabilize the Fourer probe phase nor fixing the corner intensity problem

## [v0.1.0-alpha3.9] - 2024-05-16
### Added
- Add selective figure saving with `fig_list` to `plot_summary`
- Add `pphase_smooth` smooth with Fourier probe phase unwrapping and Gaussian smooth in hope of stabilizing the Fourier probe phase during optimization (doesn't help, will be removed by the next commit)
- Add an image overlay argument and dashed lines to `plot_sigmoid_mask` for better visualization
### Changed
- Rename `scan_flip` field in `exp_params` to `scan_flipT` to be consistent with `cbed_flipT` and the PtychoShelves convention of [flipup, fliplr, transpose]
- Update all params files to be consistent with the new `scan_flipT` field
### Removed
- Remove `fix_probe_phi` constraint for simplicity because it's only for visualization purpose (setting phase at k(0,0) as 0) without actual funcitonality

## [v0.1.0-alpha3.8] - 2024-05-10
### Added
= Add `loss_simlar` for omode similarity regularization
### Changed
- Update params files with the new `obj_tilts` field
- Move `omode_occu` from `source_params` to `exp_params` for params files because `omode_occu` is no longer planned to be an optimizable/loadable variable

## [v0.1.0-alpha3.7] - 2024-05-09
### Added
- Add `get_date` and `prefix` to `make_output_folder`
### Changed
- Add new fields like `tilt_obj`, `dk`, and `dx` to `make_save_dict`
- Fix the condition statements for `self.tilt_obj` in `models` so it behaves correctly when the initialized `obj_tilts` is [0,0] so no need for tilt propagator
- Fix `kz_filter` to use `torch.real()` for the Fourier filtered obj to respect possible negative values
- Delete the just-added fftshift/ifftshift in `forward` as they don't seem to matter and the forward pass is 50% slower with additional fftshifts
### Removed
- Remove `fix_probe_com` constraints because constantly shifting global probe would cause instability of the obj. Once the obj is fairly reconstructed, shifting the global probe would require shifting the already reconstructed object (or equivalently the entire (N,2) `probe_pos_shifts`) 
- Remove `probe_mask_r` constraints because it's just not physical and I can't justify myself clipping so much real-space probe tails
- Remove `obj_chgflip` constraints because the performance is rather poor
- Remove `obj_same` constraints because the usecase is rather limited and could probably be replaced with `kz_filter`. Optimizing the broadcasted averaged slice wouldn't be computationally faster. Besides, the "rough" result cann't be used to estimate obj tilts so is less usefull than the obj regularized by `kz_filter`

## [v0.1.0-alpha3.6] - 2024-05-07
### Added
- Add `opt_obj_tilts` for global (1,2) or local/adaptive (N,2) crystal tilt correction during ptycho reconstruction
- Add `obj_tilts` fields into `exp_params` for params files
- Add `test_constraint_fn` to roughly demonstrate the `constraint_fn`
- Extend `plot_scan_pos` with an `tilts` argument so we can temporarily borrow the `plot_scan_pos` to plot out obj tilt vectors (We should decouple this in the future)
- Add `get_rbf` to roughly estimate the radius of bright-field disk of the CBEDs for `probe_k_mask`
### Changed
- Move `omode_occu` to from `source_params` to `exp_params` in `initialization` for consistency with the obj tilts as they're specific for PtyRAD and behaves differently than obj, probe, or pos

## [v0.1.0-alpha3.5] - 2024-05-06
### Added
- Add `fix_probe_phi` for Fourier probe phase constraint. This is mostly for visualizaiton purpose, not sure if it stabilize the probe update for corner intensity artifact or not
- Add `fftshift2` and `ifftshift2` to `utils` as alias functions to remove the annoying `dim=(-2,-1)` argument for most fft, ifft calls
### Changed
- Global check for the fft/ifft/fftshift/ifftshift consistency. fft is for real->k, while fftshift is for corner->center. Do ifftshift to preshift before fft. All functions except the `imshift_batch` and `imshift_single` follow this pattern
- Simplify `near_field_evolution` by removing the unused far-field propagation and unused variables

## [v0.1.0-alpha3.4] - 2024-05-04
### Added
- Add `recenter_cbeds` field into the `models` so we may shift the CBEDs based on their CoM
### Changed
- Extend `get_center_of_mass` for 3D input (N,Ny,Nx)
- Rename `imshift` into `imshift_single` and modify it in accordance with `get_center_of_mass` 

## [v0.1.0-alpha3.3] - 2024-05-04
### Added
- Add probe constraints `fix_probe_com`, `probe_mask_r`, `probe_mask_k`  in hope of fixing the probe artifact
- Add corresponding visualization function `plot_sigmoid_mask`
### Changed
- Modify the `make_output_folder` accordingly for the probe mask constraints

## [v0.1.0-alpha3.2] - 2024-05-03
### Added
- Add a simple charge-flipping constraint `obj_chgflp` for fun and hoping it might fix the "black hole atom" artifact but it does not

## [v0.1.0-alpha3.1] - 2024-05-02
### Changed
- Delete the redundant intermediate wrapper function `batch_update` as it's probably unnecessarily complicated
### Removed
- Refactor `optimization` by removing redundant losses like `loss_tv`, `loss_postiv`, and `loss_obja1`. `loss_tv` is introducing very strong mosaic-style artifact. `loss_postiv` is overlapping with the `objp_positv` constraint and we can make the constraint soft by the mixing param so there's no need for `loss_postiv`. Same reason applies to `loss_obja1` as we already have the constraint version as `obja_thresh`
- Remove the placeholder `ortho_omode` function because I don't think orthogonalizing omode is the right thing to do. The omode are not necessarily orthogonal with each other and we should probably do the decomposition in the post-processing stage, not during the reconstruction
- Remove the for-loop implementation of `orthogonalize_modes_loop` because it's no longer needed, the vectorized version is much faster

## [v0.1.0-alpha3.0] - 2024-05-01
### Added
- Add amplitude options for obj specific constraints `kz_filter` and `obj_blur` so that we have finer control
- Add obja-related loss and constraints for completeness including `loss_obja1`, `obja_thresh`, and `obj_same`
### Changed
- Major refactor of the `optimization` by encapsulating each section of losses and constraints into individual functions for clarity 
- Combine `loss_l1`, `loss_l2` into `loss_sparse` for simplicity, while I think the LN-norm with N > 1 isn't really sparse so could be useless
- Extend `plot_forward_pass` to plot the obja(zprod) as well
- Update `make_output_folder` with oalr, obj_blur, and the kz_str
- Refine visualization functions including the "last N iters" to `plot_loss_iters` for the convergence, adding "iter" to figure title, fixing the displayed scan_pos range, and the probe power to `plot_probe_modes`

## [v0.1.0-alpha2.7] - 2024-04-28
### Added
- Extend `kz_filter` to `obja`, which turns out to be critical for thick samples like Si and now the `kz_filter` is fully recovering the `multilayer_regulation` operation in PtychoShelves
- Add Fourier probe modes saving to `plot_summary`
- Add version printing in `__init__` for clarity
### Changed
- Fix the incorrect scan position plotting when it's called in `plot_summary`
- Fix the checkerboard phase artifact in k-space for `plot_probe_modes` by adding the necessary fftshift
- Add an amplitude scaling to the displayed phase for better visualization
- Refactor `imshift_batch` so that it can handle tensors with arbitrary leading dimensions (..., Ny, Nx), which is in preparation for possible need of shifting the object (omode, Nz, Ny, Nx) for global scan affine transformation
- Rename and clarify the variables associated with `create_grids` in `models`

## [v0.1.0-alpha2.6] - 2024-04-23
### Added
- `Add plot_pos_grouping` to visualize the spatial distribution for different grouping
### Changed
- Fix the incorrect sparse batch generation due to duplicated indices in the `make_batch` function
- Fix the incorrect `imshift_batch` due to a missing ")" so the shifts were not applied correctly. It has very little effect when the sub-px shifts are <<1, but it could've been limiting the capability of pos correction

## [v0.1.0-alpha2.5] - 2024-04-21
### Added
- Add `plot_scan_positions` to `plot_summary`
- Add `init_pos` argument into `plot_scan_positions` 
### Changed
- Extend `plot_probe_modes` with real/reciprocal/amp/phase options
- Refine `plot_forward_modes` with weighted sum of omode for more correct objp visualization
- Rename all occurence of `iter` into `niter` to avoid name clashing with native python build-in iterator
- Fix `make_mixed_probe` error when pmode=1 by expanding the initial probe dim into (1,Ny,Nx) and passing only the probe[0] into `make_mixed_probe` to avoid changes inside the function

## [v0.1.0-alpha2.4] - 2024-04-21
### Added
- Add `plot_summary` to show and save reconstruction summary figures
### Changed
- Refine `plot_forward_pass` to add probe and remove redundant obj panels

## [v0.1.0-alpha2.3] - 2024-04-20
### Changed
- Rename `cbeds_flip` into `cbeds_flipsT` and make it comply with PtychoShelves format (flipup,fluplr,transpose)

## [v0.1.0-alpha2.2] - 2024-04-20
### Added
- Add `objp_blur` as a new obj constraint
- Add lr format into `make_output_folder` to save learning rate into the folder name
### Changed
- Add an if block for `make_mixed_probe` so only pmode>1 would initiate the function
- Fix the incorrect pos centering inside the affine transformation block in `init_pos` in `initializaiton`

## [v0.1.0-alpha2.1] - 2024-04-19
### Added
- Add `/scripts` for .py usage
- Add a quick CoM routine `center_of_mass` in `utils` for future CBED centering purpose
### Changed
- Move `init_variables` out of `model_params` for cleaner params file setup

## [v0.1.0-alpha2.0] - 2024-04-17
### Added
- Refactor the path generation and saving by adding `make_output_folder` and `save_results`
### Changed
- Unify grouping indices selection with `select_scan_indices`, 3 modes are available ("sparse", "compact", and "random")
- Refactor the indicies and batch making functions by removing `shuffle_batches` and `select_center_rectangle`

## [v0.1.0-alpha1.0] - 2024-04-16
### Added
- Add `init_check` into `initialization`
- Add `/inputs` to store the experimental data params
- Add position initialization with global affine transformation and flip into `initialization`
- Add `README` and `CHANGELOG`
### Removed
- Remove `/archive`

## [Initial development]
### 2024-04-14
- Add `detector_blur_std` for `models`
- Add `plot_scan_positions` and `plot_affine_transformation` to `visualization`

### 2024-04-13
- Add `fix_probe_int` constraint for `optimization` 
- Fix incorrect model saving and loading from .pt
- Modify `orthogonalize_modes_vec` for probe_int sorting and printing
- Modify `obj_ROI_grid` to on-the-fly generation to reduce memory consumption for large 4D dataset
- Add `init_cache` to reduce file loading time, especially for large .mat from PtychoShelves

### 2024-04-10
- Add `CombinedConstraint` class for iter-wise constraints including orthogonalization of pmode/omode, hard positivity clamp, and kz_filter for MS-ptycho
- Modify `orthogonalize_modes_vec` to take both 3/4D input and float/complex dtype
- Modify `make_save_dict` to accomodate constraint_params
- Add `shuffle_batches` to enhance randomness of mini-batch updates

### 2024-04-06
- Add `select_center_rectangle_indices` for partial reconstructions
- Unify batch generation funciton with "random", "compact", and "sparse"
- Modify iter/batch printing indices to start at 1

### 2024-04-04
- Reorganization of package directories
- Reformat params dicts
- Refine `initialization` class
- Move probe-related functions from `initializaiton` to `utils`

### 2024-04-02
- Major refactoring of the `initialization` class
- Add epsilon to forward model for numerical stability
- Add `make_stem_probe`, `make_mixed_probe`, and `orthogonalize_mode` functions for full reconstruction capability from simulated mixed-state probe

### 2024-03-30
- Implement the `initializaiton` class

### 2024-03-29
- Add 'ortho' kw for FFT intensity normalization
- Add `Fresnel_propagator` to move probe defocus
- Modify loss function to regularize on each object mode separately

### 2024-03-23
- Clean up archived versions class/functions, rename for clarity
- Add `set_optimizer` method into `models` so we may change the list of optimizable tensors or change their learning rates
- Split `opt_obj` into `obja` and `objp`, return opt_patches, calculate loss on obj_patches
- Implement `omode_occu` for mixed object reconstruction

### 2024-03-21
- Vectorzied object cropping and Fourier shift porbes. Got probably a 10x speed up

### 2024-03-20
- Implement the `imshift` with Fourier shift

### 2024-03-18
- Initialize the GitHub repo for PtyRAD

### 2024-03-16
- Implement a probe shift with `torchviosn.transformation.affine` but it seems to introduce artifacts to the probe

### 2024-03-14
- Implement the probe position with STN (spatial transformation network) on object
- Implement object losses including obj phase L1, obj phase positivity, obj phase TV

### 2024-03-11
- Implement the mixed object reconstruciton with unified dimension (omode,Nz,Ny,Nx)

### 2024-03-10
- Implement mixed probe reconstruction with 2D/3D object

### 2024-03-08
- Start working on the PtyRAD package
