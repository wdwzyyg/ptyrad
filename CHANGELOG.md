# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [TODO]
- Add `plot_obj_fft` to `visualization` and maybe to `plot_summary` and `save_reuslts` as well. Some windowed log(S) diffractogram or P+S decomposition could be needed. (http://www.roberthovden.com/tutorial/2015_fftartifacts.html)
- Add `plot_cbeds` to `visualization`. Should take 4D cbeds input and let us plot a single CBED or pacbed based on indices with display option dp_pow.
- Add `plot_obj_tilts` to `visualization`. Decouple the functionality from `plot_scan_pos`, should have options of interpolated tilt_y, tilt_x (doing griddata interp is probably faster than scatter plot of 65k dots) or the quiver plot form
- Reimplement the `gaussian_blur` used in `optimization` for `obj_blur` constraint into `gaussian_blur3d` with a Conv3d operation and a pre-weighted Gaussian kernel so we can apply z-blur as a z-regularization, which should solve the wrap-around issue of `kz_filter`. (https://stackoverflow.com/questions/67633879/implementing-a-3d-gaussian-blur-using-separable-2d-convolutions-in-pytorch)

## [v0.1.0-beta1.0] - 2024-05-23
### Added
- Add `blur_std` option for `loss_simlar` so we could compare the std between Gaussian blurred object modes
- Add `kr_filter` as additional obj constraint to `optimization` so we could set our desirable maximum kr that contributes to the object image. Note that the positivity clipping could be doubling the apparent spatial frequencies
- Add `run_ptyrad_altas.py`, `run_ptyrad_local.py`, and `slurm_run_ptyrad.sub` as demo scripts
- Add `docs/` and `params_description.md` for some more explanation
### Change
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
- Refactor `imshift_batch` so that it can handle tensors with arvitrary leading dimensions (..., Ny, Nx), which is in preparation for possible need of shifting the object (omode, Nz, Ny, Nx) for global scan affine transformation
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
