## Define the Initialization class to initialize 4D-STEM data, object, probe, probe positions, tilts, and other variables

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ptyrad.data_io import load_fields_from_mat, load_hdf5, load_pt, load_raw, load_tif
from ptyrad.utils import (
    compose_affine_matrix,
    get_default_probe_simu_params,
    get_rbf,
    kv2wavelength,
    make_mixed_probe,
    make_stem_probe,
    near_field_evolution,
    vprint,
)

class Initializer:
    def __init__(self, exp_params, source_params, verbose=True):
        self.init_params = {'exp_params':exp_params.copy(), 'source_params':source_params} # Note that self.init_params is a copy of exp_params so they could have different values 
        self.init_variables = {}
        self.verbose=verbose
        
    def set_use_cached_flags(self, source):
        """ Set the flags for each field whether we can cache or not """
        
        source_params   = self.init_params['source_params']
        obj_source      = source_params['obj_source']
        obj_params      = source_params['obj_params']
        probe_source    = source_params['probe_source']
        probe_params    = source_params['probe_params']
        pos_source      = source_params['pos_source']
        pos_params      = source_params['pos_params']
        
        # Check if obj, probe, and pos sources are the same
        if all(field == source for field in (obj_source, probe_source, pos_source)):
            if (obj_params == probe_params == pos_params):
                self.use_cached_obj = True
                self.use_cached_probe = True
                self.use_cached_pos = True
                self.cache_path = obj_params
                self.cache_source = obj_source

        if all(field == source for field in (obj_source, probe_source)):
            if (obj_params == probe_params):
                self.use_cached_obj = True
                self.use_cached_probe = True
                self.cache_path = obj_params
                self.cache_source = obj_source


        if all(field == source for field in (obj_source, pos_source)):
            if (obj_params == pos_params):
                self.use_cached_obj = True
                self.use_cached_pos = True
                self.cache_path = obj_params
                self.cache_source = obj_source
                
        if all(field == source for field in (probe_source, pos_source)):
            if (probe_params == pos_params):
                self.use_cached_probe = True
                self.use_cached_pos = True
                self.cache_path = probe_params
                self.cache_source = probe_source
    
    def init_cache(self):
        """ Check if the source paths are the same, if so, we may cache that field to reduce file loading time """
        # Note:
        # For caching, at least 2 out of 3 fields are using the same file path
        # Therefore, there's only one possible source for the self.cache_contents
        # With 2 file source posibilities, the self.cache_contents is either caching from 'PtyRAD' or 'PtyShv'
        # Even we add more file type supports in the future (py4dstem or ptypy), the cache would still be a single file type
        
        vprint("\n### Initializing cache ###", verbose=self.verbose)
        
        # Initialize flags for cached fields
        self.use_cached_obj = False
        self.use_cached_probe = False
        self.use_cached_pos = False
        
        for source in ('PtyShv', 'PtyRAD'):
            self.set_use_cached_flags(source)
            
        if any([self.use_cached_obj, self.use_cached_probe, self.use_cached_pos]):
            if self.cache_source == 'PtyRAD':
                vprint(f"Loading 'PtyRAD' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_pt(self.cache_path)
            elif self.cache_source == 'PtyShv':
                vprint(f"Loading 'PtyShv' file from {self.cache_path} for caching", verbose=self.verbose)
                self.cache_contents = load_fields_from_mat(self.cache_path, ['object', 'probe', 'outputs.probe_positions'])
            else:
                raise KeyError(f"File type {source} not implemented for caching yet, please use 'PtyRAD', or 'PtyShv'!")
        vprint(f"use_cached_obj   = {self.use_cached_obj}", verbose=self.verbose)
        vprint(f"use_cached_probe = {self.use_cached_probe}", verbose=self.verbose)
        vprint(f"use_cached_pos   = {self.use_cached_pos}", verbose=self.verbose)
    
    def init_exp_params(self):
        vprint("\n### Initializing exp_params ###", verbose=self.verbose)
        exp_params = self.init_params['exp_params']    
        vprint("Input values are displayed below:", verbose=self.verbose)   
        for key, value in exp_params.items():
            vprint(f"{key}: {value}", verbose=self.verbose)
            
        voltage     = exp_params['kv']
        wavelength  = kv2wavelength(voltage)
        conv_angle  = exp_params['conv_angle']
        Npix        = exp_params['Npix']
        N_scan_slow = exp_params['N_scan_slow']
        N_scan_fast = exp_params['N_scan_fast']
        N_scans     = N_scan_slow * N_scan_fast
        dx          = exp_params['dx_spec']
        dk          = dk = 1/(dx*Npix)
        
        # Print some derived values for sanity check
        if self.verbose:
            print("\nDerived values given input exp_params:")
            print(f'kv          = {voltage} kV')    
            print(f'wavelength  = {wavelength:.4f} Ang')
            print(f'conv_angle  = {conv_angle} mrad')
            print(f'Npix        = {Npix} px')
            print(f'dk          = {dk:.4f} Ang^-1')
            print(f'kMax        = {(Npix*dk/2):.4f} Ang^-1')
            print(f'alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
            print(f'dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
            print(f'Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
            print(f'Real space probe extent = {dx*Npix:.4f} Ang')
        self.init_variables['Npix']        = Npix
        self.init_variables['N_scan_slow'] = N_scan_slow
        self.init_variables['N_scan_fast'] = N_scan_fast
        self.init_variables['N_scans']     = N_scans
        self.init_variables['dx']          = dx #   Ang
        self.init_variables['dk']          = dk # 1/Ang
        
    def init_measurements(self):
        source = self.init_params['source_params']['measurements_source']
        params = self.init_params['source_params']['measurements_params']
        vprint(f"\n### Initializing measurements from '{source}' ###", verbose=self.verbose)
        
        # Load file
        if source   == 'custom':
            meas = params
        elif source == 'tif':
            meas = load_tif(params['path']) # key is ignored because it's not needed for tif files
        elif source == 'mat':
            meas = load_fields_from_mat(params['path'], params['key'])[0]
        elif source == 'hdf5':
            meas = load_hdf5(params['path'], params['key'])
        elif source == 'raw':
            default_shape = (self.init_variables['N_scans'], self.init_variables['Npix'], self.init_variables['Npix'])
            meas = load_raw(params['path'], shape=params.get('shape', default_shape), offset=params.get('offset', 0), gap=params.get('gap', 1024))
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'tif', 'mat', or 'hdf5'!!")
        vprint(f"Imported meausrements shape = {meas.shape}", verbose=self.verbose)
        vprint(f"Imported meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        
        # Permute, reshape, and flip
        if self.init_params['exp_params']['meas_permute'] is not None:
            permute_order = self.init_params['exp_params']['meas_permute']
            vprint(f"Permuting measurements with {permute_order}", verbose=self.verbose)
            meas = meas.transpose(permute_order)
            
        if self.init_params['exp_params']['meas_reshape'] is not None:
            meas_shape = self.init_params['exp_params']['meas_reshape']
            vprint(f"Reshaping measurements into {meas_shape}", verbose=self.verbose)
            meas = meas.reshape(meas_shape)
            
        if self.init_params['exp_params']['meas_flipT'] is not None:
            flipT_axes = self.init_params['exp_params']['meas_flipT']
            vprint(f"Flipping measurements with [flipup, fliplr, transpose] = {flipT_axes}", verbose=self.verbose)
            
            if flipT_axes[0] != 0:
                meas = np.flip(meas, 1)
            if flipT_axes[1] != 0:
                meas = np.flip(meas, 2)
            if flipT_axes[2] != 0:
                meas = np.transpose(meas, (0,2,1))

        # Crop out a sub-region from meas
        if self.init_params['exp_params']['meas_crop'] is not None:
            vprint(f"Reshaping measurements into {meas.shape} for cropping", verbose=self.verbose)
            meas = meas.reshape(self.init_variables['N_scan_slow'], self.init_variables['N_scan_fast'], meas.shape[-2], meas.shape[-1])

            crop_indices = np.array(self.init_params['exp_params']['meas_crop'])
            vprint(f"Cropping measurements with [N_slow, N_fast, ky, kx] = {crop_indices}", verbose=self.verbose)
            Nslow_i, Nslow_f = crop_indices[0]
            Nfast_i, Nfast_f = crop_indices[1]
            ky_i,    ky_f    = crop_indices[2]
            kx_i,    kx_f    = crop_indices[3]
            meas = meas[Nslow_i:Nslow_f, Nfast_i:Nfast_f, ky_i:ky_f, kx_i:kx_f]
            vprint(f"Cropped measurements have shape (N_slow, N_fast, ky, kx) = {meas.shape}", verbose=self.verbose)
            
            # Update self.init_params['exp_params'], note that this wouldn't update the initial `exp_params` and the original exp_params will be saved into .pt
            # The updated self.init_params['exp_params'] is for initialization purpose only
            vprint("Update `exp_params` (dx_spec, Npix, N_scans, N_scan_slow, N_scan_fast) after the measurements cropping", verbose=self.verbose)
            self.init_params['exp_params']['dx_spec'] = self.init_params['exp_params']['dx_spec'] * self.init_params['exp_params']['Npix'] / meas.shape[-1]
            self.init_params['exp_params']['Npix'] = meas.shape[-1]
            self.init_params['exp_params']['N_scans'] = meas.shape[0] * meas.shape[1]
            self.init_params['exp_params']['N_scan_slow'] = meas.shape[0]
            self.init_params['exp_params']['N_scan_fast'] = meas.shape[1]
            self.init_exp_params()
            meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
            vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)
            
        # Resample diffraction patterns along the ky, kx dimension
        if self.init_params['exp_params']['meas_resample'] is not None:
            zoom_factors = np.array([1, *self.init_params['exp_params']['meas_resample']]) # scipy.ndimage.zoom applies to all axes
            meas = zoom(meas, zoom_factors, order=1)
            vprint("Update `exp_params` (Npix) after the measurements resampling", verbose=self.verbose)
            self.init_params['exp_params']['Npix'] = meas.shape[-1]
            self.init_exp_params()
            vprint(f"Resampled measurements have shape (N_scans, ky, kx) = {meas.shape}", verbose=self.verbose)
            
        # Add source size (partial spatial coherence)
        if self.init_params['exp_params']['meas_add_source_size'] is not None:
            vprint(f"Reshaping measurements into {meas.shape} for adding partial spatial coherence (source size)", verbose=self.verbose)
            meas = meas.reshape(self.init_variables['N_scan_slow'], self.init_variables['N_scan_fast'], meas.shape[-2], meas.shape[-1])
            source_size_std_ang = self.init_params['exp_params']['meas_add_source_size']
            source_size_std_px = source_size_std_ang / self.init_params['exp_params']['scan_step_size'] # The source size blur std is now in unit of scan steps
            meas = gaussian_filter(meas, sigma=source_size_std_ang, axes=(0,1)) # Partial spatial coherence is approximated by mixing DPs at nearby probe positions
            vprint(f"Adding source size (partial spatial coherence) of Gaussian blur std = {source_size_std_px:.4f} scan_step sizes or {source_size_std_ang:.4f} Ang to measurements along the scan directions", verbose=self.verbose)
            meas = meas.reshape(-1, meas.shape[-2], meas.shape[-1])
            vprint(f"Reshape measurements back to (N, ky, kx) = {meas.shape}", verbose=self.verbose)
        
        # Add detector blur (point-spread function of the detector)
        if self.init_params['exp_params']['meas_add_detector_blur'] is not None:
            detector_blur_std = self.init_params['exp_params']['meas_add_detector_blur'] # The detector blur std is in unit of final detector px
            meas = gaussian_filter(meas, sigma=detector_blur_std, axes=(-2,-1)) # Detector blur is essentially the Gaussian blur along ky, kx
            vprint(f"Adding detector blur (point-spread function of the detector) of Gaussian blur std = {detector_blur_std:.4f} px to measurements along the ky, kx directions", verbose=self.verbose)
        
        # Correct negative values if any
        if (meas < 0).any():
            min_value = meas.min()
            meas -= min_value
            # Subtraction is more general, but clipping might be more noise-robust due to the inherent denoising
            vprint(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements", verbose=self.verbose)
        
        # Add Poisson noise given electron per Ang^2
        if self.init_params['exp_params']['meas_add_poisson_noise'] is not None:
            total_electron = self.init_params['exp_params']['meas_add_poisson_noise'] * self.init_params['exp_params']['scan_step_size'] **2 # Number of electron per diffraction pattern
            meas = meas / meas.sum((-2,-1))[:,None,None]
            meas = np.random.poisson(meas * total_electron)
            vprint(f"Adding Poisson noise with a total electron per diffraction pattern of {int(total_electron)}", verbose=self.verbose)
            
        # Normalizing meas
        vprint("Normalizing measurements so the averaged measurement has max intensity at 1", verbose=self.verbose)
        meas = meas / (np.mean(meas, 0).max()) # Normalizing the meas_data so that the averaged DP has max at 1. This will make each DP has max somewhere ~ 1
        meas = meas.astype('float32')
        
        # Get rbf related values
        rbf = get_rbf(meas)
        suggested_probe_mask_k_radius = 2*rbf/meas.shape[-1]
        
        # Print out some measurements statistics
        vprint(f"Radius of bright field disk             (rbf) = {rbf} px, suggested probe_mask_k radius (rbf*2/Npix) > {suggested_probe_mask_k_radius:.2f}", verbose=self.verbose)
        vprint(f"meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})", verbose=self.verbose)
        vprint(f"measurements                      (N, Ky, Kx) = {meas.dtype}, {meas.shape}", verbose=self.verbose)
        self.init_variables['measurements'] = meas
        
    def init_probe(self):
        source = self.init_params['source_params']['probe_source']
        params = self.init_params['source_params']['probe_params']
        vprint(f"\n### Initializing probe from '{source}' ###", verbose=self.verbose)

        # Load file
        if source   == 'custom':
            probe = params
        elif source == 'PtyRAD':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_probe else load_pt(pt_path)
            probe = ckpt['optimizable_tensors']['probe'].detach().cpu().numpy()
        elif source == 'PtyShv':
            mat_path = params
            probe = self.cache_contents[1] if self.use_cached_probe else load_fields_from_mat(mat_path, 'probe')[0] # PtychoShelves probe generally has (Ny,Nx,pmode,vp) dimension. Usually people prefer pmode over vp.
            vprint(f"Input PtyShv probe has original shape {probe.shape}", verbose=self.verbose)
            if probe.ndim == 4:
                vprint(f"Import only the 1st variable probe mode to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose) # I don't find variable probe modes are particularly useful for electon ptychography
                probe = probe[...,0]
            elif probe.ndim == 2:
                vprint(f"Expanding PtyShv probe dimension to make a final probe with (pmode, Ny, Nx)", verbose=self.verbose)
                probe = probe[...,None]
            else:
                probe = probe # probe = (pmode, Ny, Nx)
            vprint("Permuting PtyShv probe into (pmode, Ny, Nx)", verbose=self.verbose) # For PtychoShelves input, do the transpose
            probe = probe.transpose(2,0,1)
        elif source == 'simu':
            probe_simu_params = params
            if probe_simu_params is None:
                vprint(f"Use exp_params and default values instead for simulation", verbose=self.verbose)
                probe_simu_params = get_default_probe_simu_params(self.init_params['exp_params'] )
            probe = make_stem_probe(probe_simu_params, verbose=self.verbose)[None,] # probe = (1,Ny,Nx) to be comply with PtyRAD convention
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
            meas = self.init_variables['measurements']
        except KeyError:
            # If 'measurements' doesn't exist, initialize it
            vprint("Warning: 'measurements' key not found. Initializing measurements for probe intensity normalization...", verbose=self.verbose)
            self.init_measurements()
            meas = self.init_variables['measurements']
            
        # Select pmode range and print summary
        probe = probe[:pmode_max]
        probe = probe / (np.sum(np.abs(probe)**2)/np.sum(meas)*len(meas))**0.5 # Normalizing the probe_data so that the sum(|probe_data|**2) is the same with an averaged single DP
        probe = probe.astype('complex64')
        vprint(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}", verbose=self.verbose)
        vprint(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.2f}, while sum(meas)/len(meas) = {np.sum(meas)/len(meas):.2f}", verbose=self.verbose)
        self.init_variables['probe'] = probe
   
    def init_pos(self):
        source          = self.init_params['source_params']['pos_source']
        params          = self.init_params['source_params']['pos_params']
        dx_spec         = self.init_params['exp_params']['dx_spec']
        scan_step_size  = self.init_params['exp_params']['scan_step_size']
        scan_rand_std   = self.init_params['exp_params']['scan_rand_std']
        N_scan_slow     = self.init_params['exp_params']['N_scan_slow']
        N_scan_fast     = self.init_params['exp_params']['N_scan_fast']
        probe_shape     = np.array([self.init_params['exp_params']['Npix']]*2)
        vprint(f"\n### Initializing probe pos from '{source}' ###", verbose=self.verbose)

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
            mat_contents = self.cache_contents if self.use_cached_pos else load_fields_from_mat(mat_path, ['object', 'probe', 'outputs.probe_positions'])
            probe_positions = mat_contents[2]
            probe_shape = mat_contents[1].shape[:2]   # Matlab probe is (Ny,Nx,pmode,vp)
            obj_shape   = mat_contents[0].shape[:2]   # Matlab object is (Ny, Nx, Nz) or (Ny,Nx)
            pos_offset = np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) - 1 # For Matlab - Python index shift
            probe_positions_yx   = probe_positions[:, [1,0]] # The first index after shifting is the row index (along vertical axis)
            pos                  = probe_positions_yx + pos_offset 
        elif source == 'simu':
            vprint(f"Simulating probe positions with dx_spec = {dx_spec}, scan_step_size = {scan_step_size}, N_scan_fast = {N_scan_fast}, N_scan_slow = {N_scan_slow}", verbose=self.verbose)
            pos = scan_step_size / dx_spec * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
            pos = pos - (pos.max(0) - pos.min(0))/2 + pos.min(0) # Center scan around origin
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
            
    def init_obj(self):
        source     = self.init_params['source_params']['obj_source']
        params     = self.init_params['source_params']['obj_params']
        dx_spec    = self.init_params['exp_params']['dx_spec']
        z_distance = self.init_params['exp_params']['z_distance']
        vprint(f"\n### Initializing obj from '{source}' ###", verbose=self.verbose)
        
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
            obj = self.cache_contents[0] if self.use_cached_obj else load_fields_from_mat(mat_path, 'object')[0]
            vprint("Expanding PtyShv object dimension", verbose=self.verbose)
            vprint(f"Input PtyShv obj has original shape {obj.shape}", verbose=self.verbose)
            if len(obj.shape) == 2: # Single-slice ptycho
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3: # MS-ptycho
                obj = obj[None,].transpose(0,3,1,2)
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
        vprint(f"object extent                 (Z, Y, X) (Ang) = {np.round((obj.shape[1]*z_distance, obj.shape[2]*dx_spec, obj.shape[3]*dx_spec),4)}", verbose=self.verbose)
        self.init_variables['obj'] = obj
                    
    def init_omode_occu(self):
        occu_type = self.init_params['exp_params']['omode_init_occu']['occu_type']
        init_occu = self.init_params['exp_params']['omode_init_occu']['init_occu']
        vprint(f"\n### Initializing omode_occu from '{occu_type}' ###", verbose=self.verbose)

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
        
    def init_H(self):
        vprint("\n### Initializing H (Fresnel propagator) ###", verbose=self.verbose)
        probe_shape = np.array([self.init_params['exp_params']['Npix']]*2) 
        z_distance = self.init_params['exp_params']['z_distance']
        lambd = kv2wavelength(self.init_params['exp_params']['kv'])
        dx_spec = self.init_params['exp_params']['dx_spec']
        extent = dx_spec * probe_shape
        vprint(f"Calculating H with probe_shape = {probe_shape}, z_distance = {z_distance:.4f} Ang, lambd = {lambd:.4f} Ang, extent = {extent.round(4)} Ang", verbose=self.verbose)
        H = near_field_evolution(probe_shape, z_distance, lambd, extent)
        H = H.astype('complex64')
        vprint(f"H                                    (Ky, Kx) = {H.dtype}, {H.shape}", verbose=self.verbose)
        self.init_variables['z_distance'] = z_distance
        self.init_variables['H'] = H
    
    def init_obj_tilts(self):
        source     = self.init_params['source_params']['tilt_source']
        params     = self.init_params['source_params']['tilt_params']
        vprint(f"\n### Initializing obj tilts from = '{source}' ###", verbose=self.verbose)
        
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
    
    def init_check(self):
        # Although some of the input experimental parameters might not be used directly by the package
        # I think it's a good practice to check for overall consistency and remind the user to check carefully
        # While these check could be performed within the init methods and achieve early return
        # It's more readable to separate the initializaiton logic with the checking logic in this way
        
        vprint("\n### Checking consistency between input params with the initialized variables ###", verbose=self.verbose)
        
        # Check the consistency of input params with the initialized variables
        exp_params  = self.init_params['exp_params']
        Npix        = exp_params['Npix']
        Nlayer      = exp_params['Nlayer']
        N_scans     = exp_params['N_scans']
        N_scan_slow = exp_params['N_scan_slow']
        N_scan_fast = exp_params['N_scan_fast']
        
        # Initialized variables
        meas            = self.init_variables['measurements']
        probe            = self.init_variables['probe']
        crop_pos         = self.init_variables['crop_pos']
        probe_pos_shifts = self.init_variables['probe_pos_shifts']
        obj              = self.init_variables['obj']
        omode_occu       = self.init_variables['omode_occu'] 
        H                = self.init_variables['H']
        obj_tilts        = self.init_variables['obj_tilts']
        
        # Check DP shape
        if Npix == meas.shape[-2] == meas.shape[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            vprint(f"Npix, DP measurements, probe, and H shapes are consistent as '{Npix}'", verbose=self.verbose)
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
            vprint(f"obj_tilts is consistent with either 1 or N_scans", verbose=self.verbose)
        else:
            raise ValueError(f"Found inconsistency between len(obj_tilts) ({len(obj_tilts)}), 1, and N_scans({N_scans})")
        
        vprint(f"Pass the consistency check of initialized variables, initialization is done!", verbose=self.verbose)
    
    def init_all(self):
        # Run this method to initialize all
        
        self.init_cache()
        self.init_exp_params()
        self.init_measurements()
        self.init_probe()
        self.init_pos()
        self.init_obj()
        self.init_omode_occu()
        self.init_H()
        self.init_obj_tilts()
        self.init_check()
        
        return self