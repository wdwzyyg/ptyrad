## Define the Initialization class to initialize 4D-STEM data, object, probe, probe positions, tilts, and other variables

from copy import deepcopy

import numpy as np
from scipy.io.matlab import matfile_version as get_matfile_version
from scipy.ndimage import gaussian_filter, zoom

from ptyrad.load import load_fields_from_mat, load_hdf5, load_npy, load_pt, load_raw, load_tif
from ptyrad.utils import (
    compose_affine_matrix,
    create_one_hot_mask,
    exponential_decay,
    fit_background,
    get_default_probe_simu_params,
    get_EM_constants,
    get_rbf,
    make_fzp_probe,
    make_mixed_probe,
    make_stem_probe,
    near_field_evolution,
    power_law,
    safe_get_nested,
    vprint,
    vprint_nested_dict,
)


class Initializer:
    def __init__(self, init_params, verbose=True):
        
        # A deepcopy creates a new object so modifying self.init_params won't affect the original init_params dict that was outside the class
        # This is important because self.init_params might get updated if there's cropping, padding, or resampling of the measurements
        # The original params file will could be directly saved to the output dir with `save.copy_params_to_dir`,
        # while we also keep a digital copy of the original params in self.init_params_original
        
        self.init_params = deepcopy(init_params) # This is the central params dict that will be used for initialization
        self.init_params_original = deepcopy(init_params)
        self.init_variables = {} # This dict stores all the variables that will be used for the later ptychography reconstruction
        self.verbose=verbose
        self.init_params_dict()
        
    def set_use_cached_flags(self, source):
        """ Set the flags for each field whether we can cache or not """
        
        obj_source  = self.init_params['obj']['source']
        obj_input   = self.init_params['obj']['input']
        probe_source = self.init_params['probe']['source']
        probe_input  = self.init_params['probe']['input']
        pos_source   = self.init_params['pos']['source']
        pos_input    = self.init_params['pos']['input']
        
        triplets = [
        ('obj', obj_source, obj_input),
        ('probe', probe_source, probe_input),
        ('pos', pos_source, pos_input)]
        
        # Helper for comparison
        def same_source_and_input(a, b):
            return a[1] == b[1] == source and a[2] == b[2]
        
        # Check if obj, probe, and pos sources are the same
        if same_source_and_input(triplets[0], triplets[1]) and same_source_and_input(triplets[1], triplets[2]):
            self.use_cached_obj = self.use_cached_probe = self.use_cached_pos = True
            self.cache_path = obj_input
            self.cache_source = obj_source
            return

        if same_source_and_input(triplets[0], triplets[1]):
            self.use_cached_obj = self.use_cached_probe = True
            self.cache_path = obj_input
            self.cache_source = obj_source

        if same_source_and_input(triplets[0], triplets[2]):
            self.use_cached_obj = self.use_cached_pos = True
            self.cache_path = obj_input
            self.cache_source = obj_source

        if same_source_and_input(triplets[1], triplets[2]):
            self.use_cached_probe = self.use_cached_pos = True
            self.cache_path = probe_input
            self.cache_source = probe_source
    
    def init_cache(self):
        """ Check if the source paths are the same, if so, we may cache that field to reduce file loading time """
        # Note:
        # For caching, at least 2 out of 3 fields are using the same file path
        # Therefore, there's only one possible source for the self.cache_contents
        # With 2 file source posibilities, the self.cache_contents is either caching from 'PtyRAD' or 'PtyShv'
        # Even we add more file type supports in the future (py4dstem or ptypy), the cache would still be a single file type
        
        vprint("### Initializing cache ###", verbose=self.verbose)
        
        # Initialize flags for cached fields
        self.use_cached_obj = False
        self.use_cached_probe = False
        self.use_cached_pos = False
        
        for source in ['PtyRAD', 'PtyShv', 'py4DSTEM']:
            self.set_use_cached_flags(source)
            
        if any([self.use_cached_obj, self.use_cached_probe, self.use_cached_pos]):
            if self.cache_source == 'PtyRAD':
                vprint(f"Loading 'PtyRAD' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_pt(self.cache_path)
            elif self.cache_source == 'PtyShv':
                vprint(f"Loading 'PtyShv' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_fields_from_mat(self.cache_path, ['object', 'probe', 'outputs.probe_positions'])
            elif self.cache_source == 'py4DSTEM':
                vprint(f"Loading 'py4DSTEM' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_hdf5(self.cache_path, dataset_key=None)
            else:
                raise KeyError(f"File type {source} not implemented for caching yet, please use 'PtyRAD', or 'PtyShv'!")
        vprint(f"use_cached_obj   = {self.use_cached_obj}", verbose=self.verbose)
        vprint(f"use_cached_probe = {self.use_cached_probe}", verbose=self.verbose)
        vprint(f"use_cached_pos   = {self.use_cached_pos}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
    
    # TODO Move this method to the top of the class later
    def init_params_dict(self):
        vprint("### Initializing init_params ###", verbose=self.verbose)
        
        # TODO Check whether I need to create these params entries at all
        # Note that the self.init_params can be modified by _meas_crop and other methods
        # So we need to call this method to re-initialize the self.init_variables
        self.meas_params = self.init_params['meas']
        self.probe_params = self.init_params['probe']
        self.pos_params = self.init_params['pos']
        self.obj_params = self.init_params['obj']
        self.tilt_params = self.init_params['tilt']
        
        vprint("Input values are displayed below:", verbose=self.verbose)
        vprint_nested_dict(self.init_params, verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
        
        illumination_type = self.probe_params.get('illumination_type') or 'electron'
        if  illumination_type == 'electron':
            voltage     = self.probe_params['kv']
            wavelength  = get_EM_constants(voltage, 'wavelength')
            conv_angle  = self.probe_params['conv_angle']
            Npix        = self.meas_params['Npix']
            N_scan_slow = self.pos_params['N_scan_slow']
            N_scan_fast = self.pos_params['N_scan_fast']
            N_scans     = N_scan_slow * N_scan_fast
            dx          = self.probe_params['dx_spec']
            dk          = 1/(dx*Npix)
            
            # Print some derived values for sanity check
            if self.verbose:
                vprint("Derived values given input exp_params:")
                vprint(f'kv          = {voltage} kV')    
                vprint(f'wavelength  = {wavelength:.4f} Ang')
                vprint(f'conv_angle  = {conv_angle} mrad')
                vprint(f'Npix        = {Npix} px')
                vprint(f'dk          = {dk:.4f} Ang^-1')
                vprint(f'kMax        = {(Npix*dk/2):.4f} Ang^-1')
                vprint(f'alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
                vprint(f'dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
                vprint(f'Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
                vprint(f'Real space probe extent = {dx*Npix:.4f} Ang')

        elif illumination_type == 'xray':
            energy      = self.probe_params['energy']
            wavelength  = 1.23984193e-9 / energy
            dx          = self.probe_params['dx_spec']
            N_scan_slow = self.pos_params['N_scan_slow']
            N_scan_fast = self.pos_params['N_scan_fast']
            N_scans     = N_scan_slow * N_scan_fast
            Npix        = self.meas_params['Npix']
            dRn         = self.probe_params['dRn']
            Rn          = self.probe_params['Rn']
            D_H         = self.probe_params['D_H']
            D_FZO       = self.probe_params['D_FZP']
            Ls          = self.probe_params['Ls']
            dk          = 1/(dx*Npix)
            
            if self.verbose:
                vprint("Derived values given input exp_params:")
                vprint(f'x-ray beam energy  = {energy} keV')    
                vprint(f'wavelength         = {wavelength} m')
                vprint(f'outmost zone width = {dRn} m')
                vprint(f'Rn                 = {Rn} m')
                vprint(f'D_H                = {D_H} m')
                vprint(f'D_FZO              = {D_FZO} m')
                vprint(f'Ls                 = {Ls} m')
                vprint(f'Npix               = {Npix} px')
                vprint(f'dx                 = {dx} m')
        
        else:
            raise KeyError(f"probe_params['illumination_type'] = {illumination_type} not implemented yet, please use either 'electron' or 'xray'!")
        
        # Save general values into init_variables        
        self.init_variables['Npix']        = Npix
        self.init_variables['N_scan_slow'] = N_scan_slow
        self.init_variables['N_scan_fast'] = N_scan_fast
        self.init_variables['N_scans']     = N_scans
        self.init_variables['dx']          = dx #   Ang
        self.init_variables['dk']          = dk # 1/Ang
        vprint(" ", verbose=self.verbose)
        
    def init_measurements(self):
        vprint("### Initializing measurements ###", verbose=self.verbose)

        meas = self._load_meas()
        meas = self._process_meas(meas)

        ## We can combine these estimations into self._analyze_meas() in the future
        # For example, get_rbf, get_detector_blur, get_meas_shifts, get_conv_angle, etc.
        
        # Get rbf related values (for electron ptychography only)
        if self.probe_params.get('illumination_type') == 'electron':
            rbf = get_rbf(meas)
            suggested_probe_mask_k_radius = 2 * rbf / meas.shape[-1]
            vprint(f"Radius of bright field disk (rbf) = {rbf:.1f} px, "
                f"suggested probe_mask_k radius (rbf*2/Npix) > {suggested_probe_mask_k_radius:.2f}", verbose=self.verbose)

        meas_avg_sum = meas.mean(0).sum()
        
        pad_mode = safe_get_nested(self.init_params, ['meas', 'process', 'pad', 'mode'])
        if pad_mode == 'on_the_fly':
            padded = self.init_variables.get('on_the_fly_meas_padded')
            padded_int_sum = padded.sum() if padded is not None else 0
            vprint(f"Adjusting `meas_avg_sum` by adding {padded_int_sum:.4f} for on_the_fly meas padding", verbose=self.verbose)
            meas_avg_sum += padded_int_sum # meas_avg_sum is used to normalize the probe intensity. 
            # Because the meas could gain intensity during on_the_fly padding, 
            # we need to consider the extra intensity from the padded region here. 
            
        self.init_variables['meas_avg_sum'] = meas_avg_sum
        self.init_variables['measurements'] = meas

        # Print out some measurements statistics
        vprint(f"meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        vprint(f"measurements                      (N, Ky, Kx) = {meas.dtype}, {meas.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
    
    def _load_meas(self):
        """Load diffraction data from file or memory according to init_params['meas']."""
        
        # Validate required fields
        try:
            meas_params = self.init_params['meas']
            source = meas_params['source']
            input_params = meas_params['input']
        except KeyError as e:
            raise ValueError(f"Missing required configuration field: {e}")

        vprint(f"Loading measurements from source = '{source}'", verbose=self.verbose)

        if source == 'custom':
            meas = input_params  # assumed to already be a NumPy array
        elif source in ('tif', 'tiff'):
            meas = load_tif(input_params['path']) # key is ignored because it's not needed for tif files
        elif source == 'mat':
            meas = load_fields_from_mat(input_params['path'], input_params['key'])[0]
        elif source == 'hdf5':
            meas = load_hdf5(input_params['path'], input_params['key']).astype('float32')
        elif source == 'npy':
            meas = load_npy(input_params['path']).astype('float32')
        elif source == 'raw':
            default_shape = (
                self.init_variables['N_scans'],
                self.init_variables['Npix'],
                self.init_variables['Npix'],
            )
            meas = load_raw(
                input_params['path'],
                shape=input_params.get('shape', default_shape),
                offset=input_params.get('offset', 0),
                gap=input_params.get('gap', 1024)
            )
        else:
            raise KeyError(f"Unsupported measurement source '{source}'. "
                        "Use 'custom', 'tif', 'mat', 'hdf5', 'npy', or 'raw'.")

        vprint(f"Imported meausrements shape / dtype = {meas.shape}, dtype = {meas.dtype}", verbose=self.verbose)
        vprint(f"Imported meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        return meas
    
    def _process_meas(self, meas):
        """
        Applies all processing steps to raw loaded measurements.
        
        """
        
        # Note that _meas_correct_neg and _meas_normalization will always be executed
        # If you really want to nullify them, explictly set 
        # 'remove_neg_values': {'mode': 'subtract_value', 'value': 0}
        # 'normalization': {'mode': 'divide_const', 'const': 1}
        
        proc = safe_get_nested(self.init_params, ['meas', 'process'], default={})

        # Simple geometric operations
        meas = self._meas_permute(meas, proc.get('permute'))
        meas = self._meas_reshape(meas, proc.get('reshape'))
        meas = self._meas_flipT(meas, proc.get('flipT'))
        
        # Operations that may change the shape of the measurements
        meas = self._meas_crop(meas, proc.get('crop'))
        meas = self._meas_correct_neg(meas, proc.get('remove_neg_values')) # meas need to be positive before the padding with background fitting mode
        meas = self._meas_pad(meas, proc.get('pad'))
        meas = self._meas_resample(meas, proc.get('resample'))
        
        # Operations that add realistic factors to (simulated perfect) measurements
        meas = self._meas_add_source_size(meas, proc.get('add_source_size'))
        meas = self._meas_add_detector_blur(meas, proc.get('add_detector_blur'))
        meas = self._meas_correct_neg(meas, proc.get('remove_neg_values')) # meas need to be positive before applying poisson noise
        meas = self._meas_add_poisson_noise(meas, proc.get('add_poisson_noise'))
        
        # Final check of the measurements
        meas = self._meas_correct_neg(meas, proc.get('remove_neg_values')) 
        meas = self._meas_normalization(meas, proc.get('normalization'))

        return meas
    
    def _meas_permute(self, meas, order):
        if order is not None:
            vprint(f"Permuting measurements with order = {order}", verbose=self.verbose)
            return meas.transpose(order)
        return meas
    
    def _meas_reshape(self, meas, target_shape):
        if target_shape is not None:
            vprint(f"Reshaping measurements to shape = {target_shape}", verbose=self.verbose)
            return meas.reshape(target_shape)
        return meas

    def _meas_flipT(self, meas, flipT_axes):
        """
        Flip and transpose measurement array.
        flipT_axes: list of 3 binary/int values [flipud, fliplr, transpose]
        """
        if flipT_axes is None:
            return meas

        # Validate length
        if not isinstance(flipT_axes, (list, tuple)) or len(flipT_axes) != 3:
            raise ValueError(f"Expected flipT_axes to be a list of 3 values, got: {flipT_axes}")

        # Safely cast all entries to int
        try:
            flipT_axes = [int(v) for v in flipT_axes]
        except Exception as e:
            raise ValueError(f"flipT_axes must contain values convertible to int (0 or 1). Got: {flipT_axes}") from e

        vprint(f"Flipping measurements with [flipud, fliplr, transpose] = {flipT_axes}", verbose=self.verbose)

        if flipT_axes[0]:
            meas = np.flip(meas, axis=1)
        if flipT_axes[1]:
            meas = np.flip(meas, axis=2)
        if flipT_axes[2]:
            meas = np.transpose(meas, (0, 2, 1))

        return meas

    def _meas_crop(self, meas, crop_ranges):
        """
        Crop measurements across 4 dimensions:
        [[slow_i, slow_f], [fast_i, fast_f], [ky_i, ky_f], [kx_i, kx_f]]
        Allows any entry to be `None` to skip cropping that axis.
        Note that this method would also update the `self.init_params` and `self.init_variables`
        """
        if crop_ranges is None:
            return meas

        if len(crop_ranges) != 4:
            raise ValueError(f"Expected 4 crop ranges [N_slow, N_fast, ky, kx], got {crop_ranges}")

        # Reshape (N, ky, kx) -> (N_slow, N_fast, ky, kx)
        Nslow, Nfast = self.init_variables['N_scan_slow'], self.init_variables['N_scan_fast']
        meas = meas.reshape(Nslow, Nfast, *meas.shape[-2:])
        vprint(f"Reshaping measurements into {meas.shape} for cropping", verbose=self.verbose)

        axes_names = ['N_slow', 'N_fast', 'ky', 'kx']
        slices = []

        for i, bounds in enumerate(crop_ranges):
            if bounds is None:
                slices.append(slice(None))
            else:
                try:
                    start, stop = bounds
                    slices.append(slice(start, stop))
                    vprint(f"Cropping axis {axes_names[i]} from {start} to {stop}", verbose=self.verbose)
                except Exception as e:
                    raise ValueError(f"Invalid crop bounds for axis {axes_names[i]}: {bounds}") from e

        meas = meas[slices[0], slices[1], slices[2], slices[3]]
        vprint(f"Cropped measurements have shape (N_slow, N_fast, ky, kx) = {meas.shape}", verbose=self.verbose)

        # Update internal variables and re-init self.init_params / self.init_variables
        vprint("Update (dx_spec, Npix, N_scans, N_scan_slow, N_scan_fast) after the measurements cropping", verbose=self.verbose)
        self.init_params['probe']['dx_spec'] *= self.init_params['meas']['Npix'] / meas.shape[-1]
        self.init_params['meas']['Npix'] = meas.shape[-1]
        self.init_params['pos']['N_scans'] = meas.shape[0] * meas.shape[1]
        self.init_params['pos']['N_scan_slow'] = meas.shape[0]
        self.init_params['pos']['N_scan_fast'] = meas.shape[1]
        vprint("Calling `init_params_dict()` again to update init_params", verbose=self.verbose)
        self.init_params_dict()
        vprint(" ", verbose=self.verbose)
        meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
        vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)

        return meas  
    
    def _meas_correct_neg(self, meas, neg_cfg):
        """
        Correct negative values in the measurement array based on the specified configuration.

        Args:
            meas (numpy.ndarray): The measurement array to process.
            neg_cfg (dict): Configuration for handling negative values. Expected keys:
                - mode (str): Method to handle negative values. Options are 'clip_neg', 'subtract_min',
                'clip_value', or 'subtract_value'. Default is 'clip_neg'.
                - value (float or None): Value used for 'clip_value' or 'subtract_value' modes. Default is None.

        Returns:
            numpy.ndarray: The processed measurement array with negative values handled.
        """
        # Check if there are negative values
        if not (meas < 0).any():
            vprint("No negative values found in measurements. Skipping non-neg correction.", verbose=self.verbose)
            return meas

        # This correction is enforced even the neg_cfg is None (not provided by user)
        if neg_cfg is None:
            neg_cfg = {}

        # Extract configuration with defaults
        mode = neg_cfg.get('mode', 'clip_neg')  # Default to 'clip_neg'
        value = neg_cfg.get('value', None)  # Default to None

        vprint(f"Removing negative values in measurement with method = {mode} and value = {value}", verbose=self.verbose)

        if mode == 'subtract_min':
            min_value = meas.min()
            meas -= min_value
            value = None  # Not relevant for this mode
            vprint(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements", verbose=self.verbose)

        elif mode == 'clip_value':
            if value is None:
                raise ValueError("Mode 'clip_value' requires a non-None 'value'.")
            vprint(f"Minimum value = {meas.min():.4f}, measurements below {value} are clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas < value] = 0

        elif mode == 'subtract_value':
            if value is None:
                raise ValueError("Mode 'subtract_value' requires a non-None 'value'.")
            vprint(f"Minimum value = {meas.min():.4f}, measurements subtracted by {value} due to the positive px value constraint of measurements", verbose=self.verbose)
            meas -= value

        elif mode == 'clip_neg': # Default mode
            vprint(f"Minimum value = {meas.min():.4f}, negative values are clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas < 0] = 0
            value = None  # Not relevant for clipping

        else:
            raise ValueError(f"Unsupported mode '{mode}' for handling negative values. Use 'clip_neg', 'subtract_min', 'clip_value', or 'subtract_value'.")

        # Final check in case the user specified value is not enough to remove all neg values
        if (meas < 0).any():
            vprint(f"User specified value = {value} is not enough to remove negative values, applying 0 clipping")
            vprint(f"Minimum value of {meas.min():.4f} is clipped to 0 due to the positive px value constraint of measurements", verbose=self.verbose)
            meas[meas<0] = 0

        return meas
    
    def _meas_normalization(self, meas, norm_cfg):
        """
        Normalize the measurement array based on the specified normalization mode.

        Args:
            meas (numpy.ndarray): The measurement array to normalize, shape (N, ky, kx).
            Returns:
            numpy.ndarray: The normalized measurement array.
        """
        
        # This correction is enforced even the norm_cfg is None (not provided by user)
        if norm_cfg is None:
            norm_cfg = {}
        
        norm_mode = norm_cfg.get('mode', 'max_at_one')  # Default to 'max_at_one'
        norm_const = norm_cfg.get('const', None)  # Used for 'divide_const' mode

        vprint(f"Normalizing measurements with mode = '{norm_mode}' and value = '{norm_const}'", verbose=self.verbose)

        if norm_mode == 'max_at_one':
            normalization_const = meas.mean(0).max()
            vprint(f"Normalizing by max of the 2D mean pattern: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'mean_at_one':
            normalization_const = meas.mean(0).mean()
            vprint(f"Normalizing by mean of the 2D mean pattern: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'sum_to_one':
            normalization_const = meas.mean(0).sum()
            vprint(f"Normalizing by sum of the 2D mean pattern: {normalization_const:.8g}", verbose=self.verbose)

        elif norm_mode == 'divide_const':
            if norm_const is None:
                raise ValueError("Mode 'divide_const' requires a non-None 'norm_const'.")
            normalization_const = norm_const
            vprint(f"Normalizing by user-defined constant: {normalization_const:.8g}", verbose=self.verbose)

        else:
            raise ValueError(f"Unsupported normalization mode '{norm_mode}'. Use 'max_at_one', 'mean_at_one', 'sum_to_one', or 'divide_const'.")

        # Normalize the measurements
        meas = meas / normalization_const
        meas = meas.astype('float32')
        return meas
    
    def _meas_pad(self, meas, pad_cfg):
        """
        _meas_pad Padd the 3D measurements array to a target size using the specified padding mode and type.

        Args:
            meas (numpy.ndarray): The measurement array to normalize, shape (N, ky, kx).
            pad_cfg (dict): A dictionary containing the padding configuration. Expected keys:
            pad_cfg = {'mode': 'on_the_fly', 'padding_type': 'power', 'target_Npix': 256, 'value': 0}

        Raises:
            KeyError: _description_
            KeyError: _description_

        Returns:
           numpy.ndarray: The padded measurement array.
        """
        
        if pad_cfg is None or pad_cfg.get('mode') is None:
            self.init_variables['on_the_fly_meas_padded'] = None
            self.init_variables['on_the_fly_meas_padded_idx'] = None
            return meas

        mode = pad_cfg['mode']  # 'precompute' or 'on_the_fly'. Use `on_the_fly` to save GPU memory
        padding_type = pad_cfg['padding_type']
        target_Npix = pad_cfg['target_Npix']
        value = pad_cfg.get('value', 10) # For constant and linear_ramp padding
        threshold = pad_cfg.get('threshold', 70) # For exp and power padding that requires fitting a thresholded mask

        vprint(f"Padding measurements with mode='{mode}', padding_type='{padding_type}', target_Npix={target_Npix}", verbose=self.verbose)

        # Get amplitude from average DP
        meas_avg = meas.mean(axis=0)
        meas_int_sum = meas_avg.sum()
        amp_avg = np.sqrt(meas_avg)
        H, W = amp_avg.shape
        
        # Calculate padding for each dimension
        pad_y = max(0, target_Npix - H)
        pad_x = max(0, target_Npix - W)
        pad_y1, pad_y2 = pad_y // 2, pad_y - pad_y // 2
        pad_x1, pad_x2 = pad_x // 2, pad_x - pad_x // 2
        pad_h1, pad_h2 = pad_y1, pad_y1 + H
        pad_w1, pad_w2 = pad_x1, pad_x1 + W

        # Create coordinate grid for radial background fitting
        y, x = np.ogrid[:target_Npix, :target_Npix]
        center = (H // 2 + pad_y1, W // 2 + pad_x1)
        r = np.sqrt((y - center[0])**2 + (x - center[1])**2) + 1e-10 # so r is never 0

        # Compute background
        if padding_type == 'constant':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant', constant_values=value)
        elif padding_type == 'edge':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='edge')
        elif padding_type == 'linear_ramp':
            amp_padded = np.pad(amp_avg, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='linear_ramp', end_values=value)
        elif padding_type == 'exp':
            mask = create_one_hot_mask(amp_avg, percentile=threshold) # It feels like we probably don't need to normalize meas before padding because the mask is calculated by percentile
            popt = fit_background(amp_avg, mask, fit_type='exp')
            amp_padded = exponential_decay(r, *popt)
        elif padding_type == 'power':
            mask = create_one_hot_mask(amp_avg, percentile=threshold)
            popt = fit_background(amp_avg, mask, fit_type='power')
            amp_padded = power_law(r, *popt)
        else:
            raise KeyError(f"Unsupported padding_type = '{padding_type}'")
        
        # Square the padded amplitude back to intensity
        meas_padded = np.square(amp_padded)[None,] # (1, ky, kx)
        meas_padded[..., pad_h1:pad_h2, pad_w1:pad_w2] = 0
        padded_int_sum = meas_padded.sum()
        vprint(f"Original meas int sum = {meas_int_sum:.4f}, padded region int sum = {padded_int_sum:.4f}, or {padded_int_sum/meas_int_sum:.2%} more intensity after padding.", verbose=self.verbose) 
        vprint("This percentage should be ideally less than 5%, or you should set a lower threshold to exclude more central region.", verbose=self.verbose)

        if mode == 'precompute':
            canvas = np.zeros((meas.shape[0], *meas_padded.shape[1:]))
            canvas += meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = meas
            meas = canvas
            self.init_variables['on_the_fly_meas_padded'] = None
            self.init_variables['on_the_fly_meas_padded_idx'] = None
        elif mode == 'on_the_fly':
            # For on_the_fly padding, we pass the padded 2D pattern (extra background) and padding indices to the model
            self.init_variables['on_the_fly_meas_padded'] = meas_padded
            self.init_variables['on_the_fly_meas_padded_idx'] = [pad_h1, pad_h2, pad_w1, pad_w2]
        else:
            raise KeyError(f"meas_pad does not support mode = '{mode}', please choose from 'on_the_fly', 'precompute', or None")

        # Update internal variables and re-init self.init_params / self.init_variables similar to _meas_crop
        vprint("Update (dx_spec, Npix, N_scans, N_scan_slow, N_scan_fast) after the measurements padding", verbose=self.verbose)
        self.init_params['probe']['dx_spec'] *= self.init_params['meas']['Npix'] / meas_padded.shape[-1]
        self.init_params['meas']['Npix'] = meas_padded.shape[-1] # This will update Npix to target_Npix no matter what mode is used
        vprint("Calling `init_params_dict()` again to update init_params", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
        self.init_params_dict()

        return meas

    def _meas_resample(self, meas, resample_cfg):
        """
        _meas_resample Resample measurements along the ky, kx dimension

        """
        if resample_cfg is None or resample_cfg.get('mode') is None:
            self.init_variables['on_the_fly_meas_scale_factors'] = None
            return meas

        # Validate required fields
        try:
            mode = resample_cfg['mode']
            Npix = self.init_params['meas']['Npix']
            scale_factors = resample_cfg['scale_factors']
        except KeyError as e:
            raise ValueError(f"Missing required configuration field: {e}")

        # Ensure scale_factors is a list or tuple of length 2
        if len(scale_factors) != 2:
            raise ValueError("scale_factors for resample must be a list or tuple of two elements.")

        if scale_factors[0] != scale_factors[1]:
            min_scale = min(scale_factors)
            vprint(f"Non-uniform scale_factors {scale_factors} detected. Using uniform scale factor: {min_scale}")
            scale_factors = [min_scale, min_scale]
        
        # If on-the-fly padding is set, force resample to be on-the-fly as well
        if self.init_variables.get('on_the_fly_meas_padded', None) is not None:
            mode = 'on_the_fly'
            vprint("'meas_resample' is set to 'on_the_fly' mode because 'meas_pad' is also set to 'on_the_fly' mode", verbose=self.verbose)

        vprint(f"Resampling measurements with mode = '{mode}', scale_factors = {scale_factors}", verbose=self.verbose)

        if mode == 'precompute':
            zoom_factors = np.array([1.0, *scale_factors]) # scipy.ndimage.zoom applies to all axes.
            meas = zoom(meas, zoom_factors, order=1) # bilinear (order=1) could prevent overshooting. Resampling would change the meas.sum(), but we have normalization at the end of the process.
            Npix = meas.shape[-1] # Update Npix
            self.init_variables['on_the_fly_meas_scale_factors'] = None

        elif mode == 'on_the_fly':
            # Don't change `meas`, just update Npix
            Npix = int(Npix * scale_factors[-1])
            self.init_variables['on_the_fly_meas_scale_factors'] = scale_factors

        else:
            raise KeyError(f"meas_resample does not support mode = '{mode}', please choose from 'on_the_fly', 'precompute', or None")

        # Update internal variables and re-init self.init_params / self.init_variables similar to _meas_crop
        self.init_params['meas']['Npix'] = Npix
        vprint(f"Update Npix into '{Npix}' after the measurements resampling", verbose=self.verbose)
        vprint(f"Resampled measurements have shape (N_scans, ky, kx) = {meas.shape}", verbose=self.verbose)
        vprint("Calling `init_params_dict()` again to update init_params", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
        self.init_params_dict()

        return meas

    def _meas_add_source_size(self, meas, source_size_std_ang):
        if source_size_std_ang is None or source_size_std_ang == 0:
            return meas

        Nslow, Nfast = self.init_variables['N_scan_slow'], self.init_variables['N_scan_fast']
        meas = meas.reshape(Nslow, Nfast, *meas.shape[-2:])
        vprint(f"Reshaping measurements into {meas.shape} for adding partial spatial coherence (source size) induced blurring on measurements", verbose=self.verbose)

        # Convert real-space blur in Angstroms to Gaussian std in scan units (px)
        source_size_std_px = source_size_std_ang / self.init_params['pos']['scan_step_size']
        vprint(f"Adding source size (partial spatial coherence) of Gaussian blur std = {source_size_std_px:.4f} scan_step sizes or {source_size_std_ang:.4f} Ang to measurements along the scan directions", verbose=self.verbose)

        # Apply blur over scan dimensions (0,1)
        meas = gaussian_filter(meas, sigma=source_size_std_px, axes=(0,1)) # Partial spatial coherence is approximated by mixing DPs at nearby probe positions
        meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
        vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)
        
        return meas

    def _meas_add_detector_blur(self, meas, detector_blur_std_px):
        """
        Add detector blur (point-spread function of the detector)

        """
        if detector_blur_std_px is None or detector_blur_std_px == 0:
            return meas
        
        meas = gaussian_filter(meas, sigma=detector_blur_std_px, axes=(-2,-1)) # Detector blur is essentially the Gaussian blur along ky, kx
        vprint(f"Adding detector blur (point-spread function of the detector) of Gaussian blur std = {detector_blur_std_px:.4f} px to measurements along the ky, kx directions", verbose=self.verbose)
        
        return meas
    
    def _meas_add_poisson_noise(self, meas, poisson_cfg):
        if poisson_cfg is None:
            return meas

        # Validate required fields
        try:
            unit = poisson_cfg['unit']
            value = poisson_cfg['value']
            scan_step_size = self.init_params['pos']['scan_step_size']
        except KeyError as e:
            raise ValueError(f"Missing required configuration field: {e}")

        # Convert units to total electrons per pattern
        if unit == 'total_e_per_pattern':
            total_electron = value
            dose = total_electron / scan_step_size**2
        elif unit == 'e_per_Ang2':
            dose = value
            total_electron = dose * scan_step_size**2
        else:
            raise ValueError(f"Unsupported unit for Poisson noise: '{unit}'. Use 'total_e_per_pattern' or 'e_per_Ang2'.")

        vprint(f"total electron per measurement = dose x scan_step_size^2 = {dose:.3f}(e-/Ang^2) x {scan_step_size:.3f}(Ang)^2 = {total_electron:.3f}", verbose=self.verbose)

        # Normalize each DP to sum = 1 before applying Poisson noise
        meas = meas / meas.sum((-2,-1))[:,None,None] # Make each slice of the meas to sum to 1
        meas = np.random.poisson(meas * total_electron)
        vprint(f"Adding Poisson noise with a total electron per diffraction pattern of {int(total_electron)}", verbose=self.verbose)
        
        return meas
        
    def init_probe(self):
        source = self.init_params['source_params']['probe_source']
        params = self.init_params['source_params']['probe_params']
        illumination_type  = self.init_params['exp_params']['illumination_type']
        vprint(f"### Initializing probe from '{source}' ###", verbose=self.verbose)

        # Load file
        if source   == 'custom':
            probe = params
        elif source == 'PtyRAD':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_probe else load_pt(pt_path)
            probe = ckpt['optimizable_tensors']['probe'].detach().cpu().numpy()
        elif source == 'PtyShv':
            mat_path = params
            mat_version = get_matfile_version(mat_path) #https://docs.scipy.org/doc/scipy-1.11.3/reference/generated/scipy.io.matlab.matfile_version.html
            use_h5py = True if mat_version[0] == 2 else False
            probe = self.cache_contents[1] if self.use_cached_probe else load_fields_from_mat(mat_path, 'probe')[0] # PtychoShelves probe generally has (Ny,Nx,pmode,vp) dimension. Usually people prefer pmode over vp.
            vprint(f"Input PtyShv probe has original shape {probe.shape}", verbose=self.verbose)
            if use_h5py:
                probe = probe.transpose(range(probe.ndim)[::-1]) 
                vprint(f"Reverse array axes because .mat (v7.3) is loaded with h5py, probe.shape = {probe.shape}", verbose=self.verbose)
            if probe.ndim == 4:
                vprint("Import only the 1st variable probe mode to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose) # I don't find variable probe modes are particularly useful for electon ptychography
                probe = probe[...,0]
            elif probe.ndim == 2:
                vprint("Expanding PtyShv probe dimension to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose)
                probe = probe[...,None]
            else:
                probe = probe # probe = (pmode, Ny, Nx)
            vprint("Permuting PtyShv probe into (pmode, Ny, Nx)", verbose=self.verbose) # For PtychoShelves input, do the transpose
            probe = probe.transpose(2,0,1)
        elif source == 'py4DSTEM':
            hdf5_path = params
            probe = self.cache_contents['probe'] if self.use_cached_probe else load_hdf5(hdf5_path, 'probe') # py4DSTEM probe generally has (pmode,Ny,Nx) dimension.
            vprint(f"Input py4DSTEM probe has original shape {probe.shape}", verbose=self.verbose)
            if probe.ndim == 2:
                vprint("Expanding py4DSTEM probe dimension to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose)
                probe = probe[None,...]
            else:
                probe = probe # probe = (pmode, Ny, Nx)
        elif source == 'simu':
            probe_simu_params = params
            if probe_simu_params is None:
                vprint("Use exp_params and default values instead for simulation", verbose=self.verbose)
                probe_simu_params = get_default_probe_simu_params(self.init_params['exp_params'])
            if illumination_type == 'electron':
                probe = make_stem_probe(probe_simu_params, verbose=self.verbose)[None,] # probe = (1,Ny,Nx) to be comply with PtyRAD convention
            elif illumination_type == 'xray':
                probe = make_fzp_probe(probe_simu_params)[None,] # simulated probe for fresnel zone plate condition
            else:
                raise KeyError(f"exp_params['illumination_type'] = {illumination_type} not implemented yet, please use either 'electron' or 'xray'!")
            if probe_simu_params['pmodes'] > 1:
                probe = make_mixed_probe(probe[0], probe_simu_params['pmodes'], probe_simu_params['pmode_init_pows'], verbose=self.verbose) # Pass in the 2D probe (Ny,Nx) to get 3D probe of (pmode, Ny, Nx)
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'PtyRAD', 'PtyShv', or 'simu'!")
        
        # Postprocess
        if self.init_params['exp_params']['probe_permute'] is not None and source != 'PtyShv':
            permute_order = self.init_params['exp_params']['probe_permute']
            vprint(f"Permuting probe with {permute_order}", verbose=self.verbose)
            probe = probe.transpose(permute_order)
            
        # Normalizing probe intensity
        pmode_max = self.init_params['exp_params']['pmode_max']
        try:
            meas_avg_sum = self.init_variables['meas_avg_sum'] # meas.mean(0).sum(), or total intensity of the averaged diffraction pattern
        except KeyError:
            # If 'measurements' doesn't exist, initialize it
            vprint("Warning: 'meas_avg_sum' key not found. Initializing measurements for probe intensity normalization...", verbose=self.verbose)
            meas_avg_sum = self.init_variables['meas_avg_sum']
            
        # Select pmode range and print summary
        probe = probe[:pmode_max]
        probe = probe / (np.sum(np.abs(probe)**2)/meas_avg_sum)**0.5 # Normalizing the probe_data so that the sum(|probe_data|**2) is the same with an averaged single DP
        probe = probe.astype('complex64')
        vprint(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}", verbose=self.verbose)
        vprint(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.2f}, while meas.mean(0).sum() = {meas_avg_sum:.2f}", verbose=self.verbose)
        self.init_variables['probe'] = probe
        vprint(" ", verbose=self.verbose)
   
    def init_pos(self):
        source          = self.init_params['source_params']['pos_source']
        params          = self.init_params['source_params']['pos_params']
        dx_spec         = self.init_params['exp_params']['dx_spec']
        scan_step_size  = self.init_params['exp_params']['scan_step_size']
        scan_rand_std   = self.init_params['exp_params']['scan_rand_std']
        N_scan_slow     = self.init_params['exp_params']['N_scan_slow']
        N_scan_fast     = self.init_params['exp_params']['N_scan_fast']
        probe_shape     = np.array([self.init_params['exp_params']['Npix']]*2)
        vprint(f"### Initializing probe pos from '{source}' ###", verbose=self.verbose)

        # Load file
        if source   == 'custom':
            pos = params
        elif source == 'PtyRAD':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_pos else load_pt(pt_path)
            crop_pos         = ckpt['model_attributes']['crop_pos'].detach().cpu().numpy()
            probe_pos_shifts = ckpt['optimizable_tensors']['probe_pos_shifts'].detach().cpu().numpy()
            pos = crop_pos + probe_pos_shifts
        elif source == 'PtyShv':
            mat_path = params
            mat_version = get_matfile_version(mat_path) #https://docs.scipy.org/doc/scipy-1.11.3/reference/generated/scipy.io.matlab.matfile_version.html
            use_h5py = True if mat_version[0] == 2 else False
            mat_contents = self.cache_contents if self.use_cached_pos else load_fields_from_mat(mat_path, ['object', 'probe', 'outputs.probe_positions'])
            if use_h5py:
                mat_contents = [arr.transpose(range(arr.ndim)[::-1]) for arr in mat_contents]
                vprint("Reverse array axes because .mat (v7.3) is loaded with h5py", verbose=self.verbose)
            probe_positions = mat_contents[2]
            probe_shape = mat_contents[1].shape[:2]   # Matlab probe is (Ny,Nx,pmode,vp)
            obj_shape   = mat_contents[0].shape[:2]   # Matlab object is (Ny, Nx, Nz) or (Ny,Nx)
            pos_offset = np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) - 1 # For Matlab - Python index shift
            probe_positions_yx   = probe_positions[:, [1,0]] # The first index after shifting is the row index (along vertical axis)
            pos                  = probe_positions_yx + pos_offset 
        elif source == 'py4DSTEM':
            hdf5_path       = params
            hdf5_contents   = self.cache_contents if self.use_cached_pos else load_hdf5(hdf5_path, 'positions_px')
            probe_positions = hdf5_contents['positions_px']
            probe_shape     = hdf5_contents['probe'].shape[-2:] # py4DSTEM probe is (pmode,Ny,Nx)
            pos             = probe_positions - np.array(probe_shape)/2 
        elif source == 'simu':
            vprint(f"Simulating probe positions with dx_spec = {dx_spec}, scan_step_size = {scan_step_size}, N_scan_fast = {N_scan_fast}, N_scan_slow = {N_scan_slow}", verbose=self.verbose)
            pos = scan_step_size / dx_spec * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
            pos = pos - pos.mean(0) # Center scan around origin
            obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
            pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift to obj coordinate
        elif source == 'foldslice_hdf5': # This preprocessing routine is equivalent to `p.src_positions='hdf5_pos';` in `fold_slice` that was used for many APS instruments
            hdf5_path = params
            ppY = load_hdf5(hdf5_path, dataset_key='ppY')
            ppX = load_hdf5(hdf5_path, dataset_key='ppX')
            pos = np.stack((-ppY, -ppX), axis=1) / dx_spec 
            pos = np.flipud(pos) # (N,2) in (pos_y_px, pos_x_px)
            obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
            pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift to obj coordinate      
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'PtyRAD', 'PtyShv', or 'simu'!")
        
        # Postprocess the scan positions if needed      
        if self.init_params['exp_params']['scan_flipT'] is not None:
            flipT_axes = np.nonzero(self.init_params['exp_params']['scan_flipT'])[0]
            if len(flipT_axes) > 0 :
                vprint(f"Flipping scan pattern (N_scan_slow, N_scan_fast, 2) with [flipup, fliplr, transpose] = {flipT_axes}", verbose=self.verbose)
                pos = pos.reshape(N_scan_slow, N_scan_fast, 2)
                pos = np.flip(pos, flipT_axes)
                pos = pos.reshape(-1,2)
            
        if self.init_params['exp_params']['scan_affine'] is not None:
            (scale, asymmetry, rotation, shear) = self.init_params['exp_params']['scan_affine']
            vprint(f"Applying affine transofrmation to scan pattern with (scale, asymmetry, rotation, shear) = {(scale, asymmetry, rotation, shear)}", verbose=self.verbose)
            pos = pos - pos.mean(0) # Center scan around origin
            pos = pos @ compose_affine_matrix(scale, asymmetry, rotation, shear)
            obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape) # Update the obj_shape
            pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift back to obj coordinate
        
        if scan_rand_std is not None:
            vprint(f"Applying Gaussian distributed random displacement with std = {scan_rand_std} px to scan positions", verbose=self.verbose)
            pos = pos + scan_rand_std * np.random.randn(*pos.shape)
        
        self.obj_extent = (1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)).astype(int)
        
        crop_pos         = np.round(pos)
        probe_pos_shifts = pos - np.round(pos)
        crop_pos         = crop_pos.astype('int16')
        probe_pos_shifts = probe_pos_shifts.astype('float32')
        
        # Print summary
        vprint(f"crop_pos                                (N,2) = {crop_pos.dtype}, {crop_pos.shape}", verbose=self.verbose)
        vprint(f"crop_pos 1st and last px coords (y,x)         = {crop_pos[0].tolist(), crop_pos[-1].tolist()}", verbose=self.verbose)
        vprint(f"crop_pos extent (Ang)                         = {(crop_pos.max(0) - crop_pos.min(0))*dx_spec}", verbose=self.verbose)
        vprint(f"probe_pos_shifts                        (N,2) = {probe_pos_shifts.dtype}, {probe_pos_shifts.shape}", verbose=self.verbose)
        self.init_variables['crop_pos'] = crop_pos
        self.init_variables['probe_pos_shifts'] = probe_pos_shifts
        self.init_variables['scan_affine'] = self.init_params['exp_params']['scan_affine']
        vprint(" ", verbose=self.verbose)
            
    def init_obj(self):
        source          = self.init_params['source_params']['obj_source']
        params          = self.init_params['source_params']['obj_params']
        dx_spec         = self.init_params['exp_params']['dx_spec']
        slice_thickness = self.init_params['exp_params']['slice_thickness']
        vprint(f"### Initializing obj from '{source}' ###", verbose=self.verbose)
        
        # Load file
        if source   == 'custom':
            obj = params
        elif source == 'PtyRAD':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_obj else load_pt(pt_path)
            obja, objp = ckpt['optimizable_tensors']['obja'].detach().cpu().numpy(), ckpt['optimizable_tensors']['objp'].detach().cpu().numpy()
            obj = obja * np.exp(1j * objp)
        elif source == 'PtyShv':
            mat_path = params
            mat_version = get_matfile_version(mat_path) #https://docs.scipy.org/doc/scipy-1.11.3/reference/generated/scipy.io.matlab.matfile_version.html
            use_h5py = True if mat_version[0] == 2 else False
            obj = self.cache_contents[0] if self.use_cached_obj else load_fields_from_mat(mat_path, 'object')[0]
            if use_h5py:
                obj = obj.transpose(range(obj.ndim)[::-1])
                vprint("Reverse array axes because .mat (v7.3) is loaded with h5py", verbose=self.verbose)
            vprint("Expanding PtyShv object dimension", verbose=self.verbose)
            vprint(f"Input PtyShv obj has original shape {obj.shape}", verbose=self.verbose)
            if len(obj.shape) == 2: # Single-slice ptycho
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3: # MS-ptycho
                obj = obj[None,].transpose(0,3,1,2)
        elif source == 'py4DSTEM':
            hdf5_path = params
            obj = self.cache_contents['object'] if self.use_cached_obj else load_hdf5(hdf5_path, 'object')
            vprint("Expanding py4DSTEM object dimension", verbose=self.verbose)
            vprint(f"Input PtyShv obj has original shape {obj.shape}", verbose=self.verbose)
            if len(obj.shape) == 2: # Single-slice ptycho
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3: # MS-ptycho
                obj = obj[None,]
        elif source == 'simu':
            if params is not None:
                obj_shape = params
            else:
                vprint("obj_shape is not provided, use exp_params, position range, and probe shape for estimated obj_shape (omode, Nz, Ny, Nx)", verbose=self.verbose)
                omode = self.init_params['exp_params']['omode_max']
                Nz    = self.init_params['exp_params']['Nlayer']
                try:
                    (Ny,Nx) = self.obj_extent
                except AttributeError:
                    vprint("Warning: 'obj_extent' field not found. Initializing positions for obj_shape estimation...", verbose=self.verbose)
                    self.init_pos()
                    (Ny,Nx) = self.obj_extent
                obj_shape = (omode, Nz, Ny, Nx)
            if len(obj_shape) != 4:
                raise ValueError(f"Input `obj_shape` = {obj_shape}, please provide a total dimension of 4 with (omode, Nz, Ny, Nx) instead!")
            else:
                obj = np.exp(1j * 1e-8*np.random.rand(*obj_shape))
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'PtyRAD', 'PtyShv', or 'simu'!")
        
        # Select omode range and print summary
        omode_max = self.init_params['exp_params']['omode_max']
        obj = obj[:omode_max].astype('complex64')
        vprint(f"object                    (omode, Nz, Ny, Nx) = {obj.dtype}, {obj.shape}", verbose=self.verbose)
        vprint(f"object extent                 (Z, Y, X) (Ang) = {np.round((obj.shape[1]*slice_thickness, obj.shape[2]*dx_spec, obj.shape[3]*dx_spec),4)}", verbose=self.verbose)
        self.init_variables['obj'] = obj
        vprint(" ", verbose=self.verbose)
                    
    def init_omode_occu(self):
        occu_type = self.init_params['exp_params']['omode_init_occu']['occu_type']
        init_occu = self.init_params['exp_params']['omode_init_occu']['init_occu']
        vprint(f"### Initializing omode_occu from '{occu_type}' ###", verbose=self.verbose)

        if occu_type   == 'custom':
            omode_occu = init_occu
        elif occu_type == 'uniform':
            omode = self.init_params['exp_params']['omode_max']
            omode_occu = np.ones(omode)/omode
        else:
            raise KeyError(f"Initialization method {occu_type} not implemented yet, please use 'custom' or 'uniform'!")
        
        omode_occu = omode_occu.astype('float32')
        vprint(f"omode_occu                            (omode) = {omode_occu.dtype}, {omode_occu.shape}", verbose=self.verbose)
        self.init_variables['omode_occu'] = omode_occu
        vprint(" ", verbose=self.verbose)
        
    def init_H(self):
        vprint("### Initializing H (Fresnel propagator) ###", verbose=self.verbose)
        probe_shape = np.array([self.init_params['exp_params']['Npix']]*2) 
        slice_thickness = self.init_params['exp_params']['slice_thickness']
        dx_spec = self.init_params['exp_params']['dx_spec']
        
        if self.init_params['exp_params']['illumination_type'] == 'electron':
            lambd = get_EM_constants(self.init_params['exp_params']['kv'], 'wavelength')
            vprint(f"Calculating H with probe_shape = {probe_shape}, dx_spec = {dx_spec:.4f} Ang, slice_thickness = {slice_thickness:.4f} Ang, lambd = {lambd:.4f} Ang", verbose=self.verbose)
        elif self.init_params['exp_params']['illumination_type'] == 'xray':
            lambd = 1.23984193e-9 / (self.init_params['exp_params']['energy'])
            vprint(f"Calculating H with probe_shape = {probe_shape}, dx_spec = {dx_spec} m, slice_thickness = {slice_thickness} m, lambd = {lambd} m", verbose=self.verbose)
        else:
            raise KeyError(f"exp_params['illumination_type'] = {self.init_params['exp_params']['illumination_type']} not implemented yet, please use either 'electron' or 'xray'!")
        
        H = near_field_evolution(probe_shape, dx_spec, slice_thickness, lambd)
        H = H.astype('complex64')
        vprint(f"H                                    (Ky, Kx) = {H.dtype}, {H.shape}", verbose=self.verbose)
        self.init_variables['lambd'] = lambd
        self.init_variables['slice_thickness'] = slice_thickness
        self.init_variables['H'] = H
        vprint(" ", verbose=self.verbose)
    
    def init_obj_tilts(self):
        source     = self.init_params['source_params']['tilt_source']
        params     = self.init_params['source_params']['tilt_params']
        vprint(f"### Initializing obj tilts from = '{source}' ###", verbose=self.verbose)
        
        if source == 'custom':
            obj_tilts = params
        elif source == 'PtyRAD':
            pt_path = params
            ckpt = self.cache_contents if pt_path == self.cache_path else load_pt(pt_path)            
            obj_tilts = np.float32(ckpt['optimizable_tensors']['obj_tilts'].detach().cpu().numpy())
            vprint(f"Initialized obj_tilts with loaded obj_tilts from PtyRAD, mean obj_tilts = {obj_tilts.mean(0).round(2)} (theta_y, theta_x) mrad", verbose=self.verbose)
        elif source == 'simu':
            N_scans    = self.init_params['exp_params']['N_scans']
            tilt_type  = params.get('tilt_type')
            init_tilts = params.get('init_tilts') 
            if tilt_type == 'each':
                obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(N_scans,2))
                vprint(f"Initialized obj_tilts with init_tilts = {init_tilts} (theta_y, theta_x) mrad", verbose=self.verbose)
            elif tilt_type == 'all':
                obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(1,2))
                vprint(f"Initialized obj_tilts with init_tilts = {init_tilts} (theta_y, theta_x) mrad", verbose=self.verbose)
            else:
                raise KeyError(f"Tilt type {tilt_type} not implemented yet, please use either 'each', or 'all' when initializing obj_tilts with 'simu'!")
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'PtyRAD', or 'simu'!")
        
        # Print summary
        self.init_variables['obj_tilts'] = obj_tilts
        vprint(f"obj_tilts                              (N, 2) = {obj_tilts.dtype}, {obj_tilts.shape}", verbose=self.verbose)
        vprint(" ", verbose=self.verbose)
    
    def init_check(self):
        # Although some of the input experimental parameters might not be used directly by the package
        # I think it's a good practice to check for overall consistency and remind the user to check carefully
        # While these check could be performed within the init methods and achieve early return
        # It's more readable to separate the initializaiton logic with the checking logic in this way
        
        vprint("### Checking consistency between input params with the initialized variables ###", verbose=self.verbose)
        
        # Check the consistency of input params with the initialized variables
        exp_params  = self.init_params['exp_params']
        Npix        = exp_params['Npix']
        Nlayer      = exp_params['Nlayer']
        N_scans     = exp_params['N_scans']
        N_scan_slow = exp_params['N_scan_slow']
        N_scan_fast = exp_params['N_scan_fast']
        
        # Initialized variables
        meas             = self.init_variables['measurements']
        probe            = self.init_variables['probe']
        crop_pos         = self.init_variables['crop_pos']
        probe_pos_shifts = self.init_variables['probe_pos_shifts']
        obj              = self.init_variables['obj']
        omode_occu       = self.init_variables['omode_occu'] 
        H                = self.init_variables['H']
        obj_tilts        = self.init_variables['obj_tilts']
        if self.init_variables.get('on_the_fly_meas_padded', None) is not None:
            target_Npix  = self.init_variables['on_the_fly_meas_padded'].shape[-1]
        else:
            target_Npix  = meas.shape[-1]
        if self.init_variables.get('on_the_fly_meas_scale_factors', None) is not None:
            scale_factors = self.init_variables['on_the_fly_meas_scale_factors']
        else:
            scale_factors = [1,1]   
        
        # Check DP shape
        if Npix == meas.shape[-2] == meas.shape[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes are consistent as '{Npix}'", verbose=self.verbose)
        elif Npix == target_Npix == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement padding", verbose=self.verbose)
        elif Npix == meas.shape[-2]*scale_factors[-2] == meas.shape[-1]*scale_factors[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement resampling", verbose=self.verbose)
        elif Npix == target_Npix*scale_factors[-2] == target_Npix*scale_factors[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes will be consistent as '{Npix}' during on-the-fly measurement padding and then resampling", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between Npix({Npix}), DP measurements({meas.shape[-2:]}), probe({probe.shape[-2:]}), and H({H.shape[-2:]}) shape")
        # Check scan pattern
        if N_scans == len(meas) == N_scan_slow*N_scan_fast == len(crop_pos) == len(probe_pos_shifts):
            vprint(f"N_scans, len(meas), N_scan_slow*N_scan_fast, len(crop_pos), and len(probe_pos_shifts) are consistent as '{N_scans}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconstency between N_scans({N_scans}), len(meas)({len(meas)}), N_scan_slow({N_scan_slow})*N_scan_fast({N_scan_fast}), len(crop_pos)({len(crop_pos)}), and len(probe_pos_shifts)({len(probe_pos_shifts)})")
        
        # Check object shape
        if obj.shape[0] == len(omode_occu):
            vprint(f"obj.shape[0] is consistent with len(omode_occu) as '{obj.shape[0]}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between obj.shape[0]({obj.shape[0]}) and len(omode_occu)({len(omode_occu)})")
        
        if obj.shape[1] == Nlayer:        
            vprint(f"obj.shape[1] is consistent with Nlayer as '{Nlayer}'", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between obj.shape[1]({obj.shape[1]}) and Nlayer({Nlayer})")

        # Check obj tilts
        if len(obj_tilts) in [1, N_scans]:
            vprint("obj_tilts is consistent with either 1 or N_scans", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between len(obj_tilts) ({len(obj_tilts)}), 1, and N_scans({N_scans})")
        
        vprint("Pass the consistency check of initialized variables, initialization is done!", verbose=self.verbose)
    
    def init_all(self):
        # Run this method to initialize all
        
        self.init_cache()
        self.init_measurements()
        self.init_probe()
        self.init_pos()
        self.init_obj()
        self.init_omode_occu()
        self.init_H()
        self.init_obj_tilts()
        self.init_check()
        
        return self