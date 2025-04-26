
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
- Decouple the BO error from reconstruction loss so we can test different setup
- We should really add a quality metrics selection so the BO can be optimized on something other than data error
- A more reasonable multi-objective optimization is probably the speed vs. quality, because multi-objective is usually more concern about trade-offs. It's a bit tricky to discern data error with reconstruction quality because they often correlate very well until certain level. Although we can combine data error with a subjective quality metric into one overall ptycho metric, like our soft regularization term (data error + sparsity loss), it's still adding an extra weighting parameter. Another valuable dimension might be the dose budget, so we're constantly compromising between dose and quality, although that's something we can only do with simulations.
- Refactor or decouple the measurements initialization from Initializer so we can have more hypertunable parameters and cleaner optuna_objective by re-initializing everything except loading measurements
## Recon workflow
- Decouple the reconstruction error with data error so that we can reconstruct with whatever target loss, while having an independent data error metric 
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
- Run through Ruff formatter for PEP8 code style (for certain part of the PtyRAD codebase)
- Add type hints
- Refine doc strings (Google style)
- Use Sphinx and Napolean for API documentation on Read the Docs
- Unified the usage of explicit key or .get for dict
- Unified meshgrid usage, naming, and unit would be nice
## PyTorch performance tuning
- Use DataLoader for measurements. This could be a bit slower than directly loading entire measurements into memory, but it allows dataset larger than memory, and also makes parallel on multiple GPUs possible
- Delete used variables for lower memory footprint
- Use in-place operations on tensors don't require grad