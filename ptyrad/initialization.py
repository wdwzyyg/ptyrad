## Initialization

import numpy as np
from .data_io import load_fields_from_mat, load_hdf5, load_pt, load_tif
from .utils import kv2wavelength, near_field_evolution, make_stem_probe, make_mixed_probe, get_default_probe_simu_params, compose_affine_matrix, get_rbf


class Initializer:
    def __init__(self, exp_params, source_params):
        self.init_params = {'exp_params':exp_params, 'source_params':source_params}
        self.init_variables = {}
    
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
        # With 2 file source posibilities, the self.cache_contents is either caching from 'pt' or 'PtyShv'
        # Even we add more file type supports in the future (py4dstem or ptypy), the cache would still be a single file type
        
        print("\n### Initializing cache ###")
        
        # Initialize flags for cached fields
        self.use_cached_obj = False
        self.use_cached_probe = False
        self.use_cached_pos = False
        
        for source in ('PtyShv', 'pt'):
            self.set_use_cached_flags(source)
            
        if any([self.use_cached_obj, self.use_cached_probe, self.use_cached_pos]):
            if self.cache_source == 'pt':
                print(f"Loading 'pt' file from {self.cache_path} for caching")
                self.cache_contents = load_pt(self.cache_path)
            elif self.cache_source == 'PtyShv':
                print(f"Loading 'PtyShv' file from {self.cache_path} for caching")
                self.cache_contents = load_fields_from_mat(self.cache_path, ['object', 'probe', 'outputs.probe_positions'])
            else:
                raise KeyError(f"File type {source} not implemented for caching yet, please use 'pt', or 'PtyShv'!")
        print(f"use_cached_obj   = {self.use_cached_obj}")
        print(f"use_cached_probe = {self.use_cached_probe}")
        print(f"use_cached_pos   = {self.use_cached_pos}")
    
    def init_exp_params(self):
        print("\n### Initializing exp_params ###")
        exp_params = self.init_params['exp_params']    
        print("Input values are displayed below:")   
        for key, value in exp_params.items():
            print(f"{key}: {value}")
            
        voltage    = exp_params['kv']
        wavelength = kv2wavelength(voltage)
        conv_angle = exp_params['conv_angle']
        Npix       = exp_params['Npix']
        dx         = exp_params['dx_spec']
        dk         = dk = 1/(dx*Npix)
        
        # Print some derived values for sanity check
        print("Derived values given input exp_params:")
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
        self.init_variables['dx'] = dx #   Ang
        self.init_variables['dk'] = dk # 1/Ang
        
            
    def init_measurements(self):
        source = self.init_params['source_params']['measurements_source']
        params = self.init_params['source_params']['measurements_params']
        print(f"\n### Initializing measurements from '{source}' ###")

        # Load file
        if source   == 'custom':
            cbeds = params
        elif source == 'tif':
            cbeds = load_tif(params)
        elif source == 'mat':
            cbeds = load_fields_from_mat(params[0], params[1])[0]
        elif source == 'hdf5':
            cbeds = load_hdf5(params[0], params[1])
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'tif', 'mat', or 'hdf5'!!")
        print(f"Imported meausrements shape = {cbeds.shape}")
        print(f"Imported meausrements int. statistics (min, mean, max) = ({cbeds.min():.4f}, {cbeds.mean():.4f}, {cbeds.max():.4f})")

        # Correct negative values if any
        if (cbeds < 0).any():
            min_value = cbeds.min()
            cbeds -= min_value
            # Subtraction is more general, but clipping might be more noise-robust due to the inherent denoising
            print(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements")
        
        # Permute, reshape, and flip
        if self.init_params['exp_params']['cbeds_permute'] is not None:
            permute_order = self.init_params['exp_params']['cbeds_permute']
            print(f"Permuting measurements with {permute_order}")
            cbeds = cbeds.transpose(permute_order)
            
        if self.init_params['exp_params']['cbeds_reshape'] is not None:
            cbeds_shape = self.init_params['exp_params']['cbeds_reshape']
            print(f"Reshaping measurements into {cbeds_shape}")
            cbeds = cbeds.reshape(cbeds_shape)
            
        if self.init_params['exp_params']['cbeds_flipT'] is not None:
            flipT_axes = self.init_params['exp_params']['cbeds_flipT']
            print(f"Flipping measurements with [flipup, fliplr, transpose] = {flipT_axes}")
            
            if flipT_axes[0] != 0:
                cbeds = np.flip(cbeds, 1)
            if flipT_axes[1] != 0:
                cbeds = np.flip(cbeds, 2)
            if flipT_axes[2] != 0:
                cbeds = np.transpose(cbeds, (0,2,1))
            
        # Normalizing cbeds
        print("Normalizing measurements so the averaged measurement has max intensity at 1")
        cbeds = cbeds / (np.mean(cbeds, 0).max()) # Normalizing the cbeds_data so that the averaged CBED has max at 1. This will make each CBED has max somewhere ~ 1
        cbeds = cbeds.astype('float32')
        
        # Get rbf related values
        rbf = get_rbf(cbeds)
        suggested_probe_mask_k_radius = 2*rbf/cbeds.shape[-1]
        
        # Print out some measurements statistics
        print(f"Radius of bright field disk             (rbf) = {rbf} px, suggested probe_mask_k radius (rbf*2/Npix) > {suggested_probe_mask_k_radius:.2f}")
        print(f"meausrements int. statistics (min, mean, max) = ({cbeds.min():.4f}, {cbeds.mean():.4f}, {cbeds.max():.4f})")
        print(f"measurements                      (N, Ky, Kx) = {cbeds.dtype}, {cbeds.shape}")
        self.init_variables['measurements'] = cbeds
        
    def init_probe(self):
        source = self.init_params['source_params']['probe_source']
        params = self.init_params['source_params']['probe_params']
        print(f"\n### Initializing probe from '{source}' ###")

        # Load file
        if source   == 'custom':
            probe = params
        elif source == 'pt':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_probe else load_pt(pt_path)
            probe = ckpt['optimizable_tensors']['probe'].detach().cpu().numpy()
        elif source == 'PtyShv':
            mat_path = params
            probe = self.cache_contents[1] if self.use_cached_probe else load_fields_from_mat(mat_path, 'probe')[0] # PtychoShelves probe generally has (Ny,Nx,pmode,vp) dimension. Usually people prefer pmode over vp.
            print(f"Input PtyShv probe has original shape {probe.shape}")
            if probe.ndim == 4:
                print(f"Import only the 1st variable probe mode to make a final probe with (pmode, Ny, Nx)") # I don't find variable probe modes are particularly useful for electon ptychography
                probe = probe[...,0]
            elif probe.ndim == 2:
                print(f"Expanding PtyShv probe dimension to make a final probe with (pmode, Ny, Nx)")
                probe = probe[...,None]
            else:
                probe = probe # probe = (pmode, Ny, Nx)
            print("Permuting PtyShv probe into (pmode, Ny, Nx)") # For PtychoShelves input, do the transpose
            probe = probe.transpose(2,0,1)
        elif source == 'simu':
            probe_simu_params = params
            if probe_simu_params is None or type(probe_simu_params) == str:
                print(f"exp_params[`probe_simu_params`] is set to `{probe_simu_params}`, use exp_params and default values instead for simulation")
                probe_simu_params = get_default_probe_simu_params(self.init_params['exp_params'] )
            probe = make_stem_probe(probe_simu_params)[None,] # probe = (1,Ny,Nx) to be comply with PtyRAD convention
            if probe_simu_params['pmodes'] > 1:
                probe = make_mixed_probe(probe[0], probe_simu_params['pmodes'], probe_simu_params['pmode_init_pows']) # Pass in the 2D probe (Ny,Nx) to get 3D probe of (pmode, Ny, Nx)
                                
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        
        # Postprocess
        if self.init_params['exp_params']['probe_permute'] is not None and source != 'PtyShv':
            permute_order = self.init_params['exp_params']['probe_permute']
            print(f"Permuting probe with {permute_order}")
            probe = probe.transpose(permute_order)
            
        # Normalizing probe intensity
        pmode_max = self.init_params['exp_params']['pmode_max']
        try:
            cbeds = self.init_variables['measurements']
        except KeyError:
            # If 'measurements' doesn't exist, initialize it
            print("Warning: 'measurements' key not found. Initializing measurements for probe intensity normalization...")
            self.init_measurements()
            cbeds = self.init_variables['measurements']
            
        # Select pmode range and print summary
        probe = probe[:pmode_max]
        probe = probe / (np.sum(np.abs(probe)**2)/np.sum(cbeds)*len(cbeds))**0.5 # Normalizing the probe_data so that the sum(|probe_data|**2) is the same with an averaged single CBED
        probe = probe.astype('complex64')
        print(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}")
        print(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.2f}, while sum(cbeds)/len(cbeds) = {np.sum(cbeds)/len(cbeds):.2f}")
        self.init_variables['probe'] = probe
   
    def init_pos(self):
        source          = self.init_params['source_params']['pos_source']
        params          = self.init_params['source_params']['pos_params']
        dx_spec         = self.init_params['exp_params']['dx_spec']
        scan_step_size  = self.init_params['exp_params']['scan_step_size']
        N_scan_slow     = self.init_params['exp_params']['N_scan_slow']
        N_scan_fast     = self.init_params['exp_params']['N_scan_fast']
        probe_shape     = np.array([self.init_params['exp_params']['Npix']]*2)
        
        print(f"\n### Initializing probe pos from '{source}' ###")

        # Load file
        if source   == 'custom':
            pos = params
        elif source == 'pt':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_pos else load_pt(pt_path)
            crop_pos         = ckpt['model_params']['crop_pos'].detach().cpu().numpy()
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
            print(f"Simulating probe positions with dx_spec = {dx_spec}, scan_step_size = {scan_step_size}, N_scan_fast = {N_scan_fast}, N_scan_slow = {N_scan_slow}")
            pos = scan_step_size / dx_spec * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
            pos = pos - (pos.max(0) - pos.min(0))/2 + pos.min(0) # Center scan around origin
            obj_shape = 1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)
            pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift to obj coordinate
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        
        # Postprocess the scan positions if needed        
        if self.init_params['exp_params']['scan_flip'] is not None:
            scan_flip = self.init_params['exp_params']['scan_flip']
            print(f"Flipping scan pattern (N_scan_slow, N_scan_fast, 2) with axes = {scan_flip}")
            pos = pos.reshape(N_scan_slow, N_scan_fast, 2)
            pos = np.flip(pos, scan_flip)
            pos = pos.reshape(-1,2)
        if self.init_params['exp_params']['scan_affine'] is not None:
            (scale, asymmetry, rotation, shear) = self.init_params['exp_params']['scan_affine']
            print(f"Applying affine transofrmation to scan pattern with (scale, asymmetry, rotation, shear) = {(scale, asymmetry, rotation, shear)}")
            pos = pos - pos.mean(0) # Center scan around origin
            pos = pos @ compose_affine_matrix(scale, asymmetry, rotation, shear)
            pos = pos + np.ceil((np.array(obj_shape)/2) - (np.array(probe_shape)/2)) # Shift back to obj coordinate
        
        self.obj_extent = (1.2 * np.ceil(pos.max(0) - pos.min(0) + probe_shape)).astype(int)
        
        crop_pos         = np.round(pos)
        probe_pos_shifts = pos - np.round(pos)
        crop_pos         = crop_pos.astype('int16')
        probe_pos_shifts = probe_pos_shifts.astype('float32')
        
        # Print summary
        print(f"crop_pos                                (N,2) = {crop_pos.dtype}, {crop_pos.shape}")
        print(f"crop_pos 1st and last px coords (y,x)         = {crop_pos[0].tolist(), crop_pos[-1].tolist()}")
        print(f"crop_pos extent (Ang)                         = {(crop_pos.max(0) - crop_pos.min(0))*dx_spec}")
        print(f"probe_pos_shifts                        (N,2) = {probe_pos_shifts.dtype}, {probe_pos_shifts.shape}")
        self.init_variables['crop_pos'] = crop_pos
        self.init_variables['probe_pos_shifts'] = probe_pos_shifts
            
    def init_obj(self):
        source     = self.init_params['source_params']['obj_source']
        params     = self.init_params['source_params']['obj_params']
        dx_spec    = self.init_params['exp_params']['dx_spec']
        z_distance = self.init_params['exp_params']['z_distance']
        print(f"\n### Initializing obj from '{source}' ###")
        
        # Load file
        if source   == 'custom':
            obj = params
        elif source == 'pt':
            pt_path = params
            ckpt = self.cache_contents if self.use_cached_obj else load_pt(pt_path)
            obja, objp = ckpt['optimizable_tensors']['obja'].detach().cpu().numpy(), ckpt['optimizable_tensors']['objp'].detach().cpu().numpy()
            obj = obja * np.exp(1j * objp)
        elif source == 'PtyShv':
            mat_path = params
            obj = self.cache_contents[0] if self.use_cached_obj else load_fields_from_mat(mat_path, 'object')[0]
            print("Expanding PtyShv object dimension")
            print(f"Input PtyShv obj has original shape {obj.shape}")
            if len(obj.shape) == 2: # Single-slice ptycho
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3: # MS-ptycho
                obj = obj[None,].transpose(0,3,1,2)
        elif source == 'simu':
            if params is not None:
                obj_shape = params
            else:
                print("obj_shape is not provided, use exp_params, position range, and probe shape for estimated obj_shape (omode, Nz, Ny, Nx)")
                omode = self.init_params['exp_params']['omode_max']
                Nz    = self.init_params['exp_params']['Nlayer']
                try:
                    (Ny,Nx) = self.obj_extent
                except AttributeError:
                    print("Warning: 'obj_extent' field not found. Initializing positions for obj_shape estimation...")
                    self.init_pos()
                    (Ny,Nx) = self.obj_extent
                obj_shape = (omode, Nz, Ny, Nx)
            if len(obj_shape) != 4:
                raise ValueError(f"Input `obj_shape` = {obj_shape}, please provide a total dimension of 4 with (omode, Nz, Ny, Nx) instead!")
            else:
                obj = np.exp(1j * 1e-8*np.random.rand(*obj_shape))
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        
        # Select omode range and print summary
        omode_max = self.init_params['exp_params']['omode_max']
        obj = obj[:omode_max].astype('complex64')
        print(f"object                    (omode, Nz, Ny, Nx) = {obj.dtype}, {obj.shape}")
        print(f"object extent                 (Z, Y, X) (Ang) = {np.round((obj.shape[1]*z_distance, obj.shape[2]*dx_spec, obj.shape[3]*dx_spec),4)}")
        self.init_variables['obj'] = obj
                    
    def init_omode_occu(self):
        occu_type = self.init_params['exp_params']['omode_init_occu']['occu_type']
        init_occu = self.init_params['exp_params']['omode_init_occu']['init_occu']
        print(f"\n### Initializing omode_occu from '{occu_type}' ###")

        if occu_type   == 'custom':
            omode_occu = init_occu
        elif occu_type == 'uniform':
            omode = self.init_params['exp_params']['omode_max']
            omode_occu = np.ones(omode)/omode
        else:
            raise KeyError(f"Initialization method {occu_type} not implemented yet, please use 'custom' or 'uniform'!")
        omode_occu = omode_occu.astype('float32')
        print(f"omode_occu                            (omode) = {omode_occu.dtype}, {omode_occu.shape}")
        self.init_variables['omode_occu'] = omode_occu
        
    def init_H(self):
        print("\n### Initializing H (Fresnel propagator) ###")
        probe_shape = np.array([self.init_params['exp_params']['Npix']]*2) 
        z_distance = self.init_params['exp_params']['z_distance']
        lambd = kv2wavelength(self.init_params['exp_params']['kv'])
        dx_spec = self.init_params['exp_params']['dx_spec']
        extent = dx_spec * probe_shape
        print(f"Calculating H with probe_shape = {probe_shape}, z_distance = {z_distance:.4f} Ang, lambd = {lambd:.4f} Ang, extent = {extent.round(4)} Ang")
        H = near_field_evolution(probe_shape, z_distance, lambd, extent)
        H = H.astype('complex64')
        print(f"H                                    (Ky, Kx) = {H.dtype}, {H.shape}")
        self.init_variables['z_distance'] = z_distance
        self.init_variables['H'] = H
    
    def init_obj_tilts(self):
        N_scans    = self.init_params['exp_params']['N_scans']
        init_tilts = self.init_params['exp_params']['obj_tilts']['init_tilts'] 
        tilt_type  = self.init_params['exp_params']['obj_tilts']['tilt_type'] 
        print(f"\n### Initializing obj tilts with tilt_type = '{tilt_type}' ###")
        if tilt_type == 'each':
            obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(N_scans,2))
        elif tilt_type == 'all':
            obj_tilts = np.broadcast_to(np.float32(init_tilts), shape=(1,2))
        else:
            raise KeyError(f"Unknown tilt_type = {tilt_type}, please use 'each' or 'all'!")
        
        print(f"Initialized obj_tilts with init_tilts = {init_tilts} (theta_y, theta_x) mrad")
        self.init_variables['obj_tilts'] = obj_tilts
        print(f"obj_tilts                              (N, 2) = {obj_tilts.dtype}, {obj_tilts.shape}")
    
    def init_check(self):
        # Although some of the input experimental parameters might not be used directly by the package
        # I think it's a good practice to check for overall consistency and remind the user to check carefully
        # While these check could be performed within the init methods and achieve early return
        # It's more readable to separate the initializaiton logic with the checking logic in this way
        
        print("\n### Checking consistency between input params with the initialized variables ###")
        
        # Check the consistency of input params with the initialized variables
        exp_params  = self.init_params['exp_params']
        Npix        = exp_params['Npix']
        Nlayer      = exp_params['Nlayer']
        N_scans     = exp_params['N_scans']
        N_scan_slow = exp_params['N_scan_slow']
        N_scan_fast = exp_params['N_scan_fast']
        
        # Initialized variables
        cbeds            = self.init_variables['measurements']
        probe            = self.init_variables['probe']
        crop_pos         = self.init_variables['crop_pos']
        probe_pos_shifts = self.init_variables['probe_pos_shifts']
        obj              = self.init_variables['obj']
        omode_occu       = self.init_variables['omode_occu'] 
        H                = self.init_variables['H']
        obj_tilts        = self.init_variables['obj_tilts']
        
        # Check CBED shape
        if Npix == cbeds.shape[-2] == cbeds.shape[-1] == probe.shape[-2] == probe.shape[-1] == H.shape[-2] == H.shape[-1]:
            print(f"Npix, CBED measurements, probe, and H shapes are consistent as '{Npix}'")
        else:
            raise ValueError(f"Found inconsistency between Npix({Npix}), CBED measurements({cbeds.shape[-2:]}), probe({probe.shape[-2:]}), and H({H.shape[-2:]}) shape")
        # Check scan pattern
        if N_scans == len(cbeds) == N_scan_slow*N_scan_fast == len(crop_pos) == len(probe_pos_shifts):
            print(f"N_scans, len(cbeds), N_scan_slow*N_scan_fast, len(crop_pos), and len(probe_pos_shifts) are consistent as '{N_scans}'")
        else:
            raise ValueError(f"Found inconstency between N_scans({N_scans}), len(cbeds)({len(cbeds)}), N_scan_slow({N_scan_slow})*N_scan_fast({N_scan_fast}), len(crop_pos)({len(crop_pos)}), and len(probe_pos_shifts)({len(probe_pos_shifts)})")
        
        # Check object shape
        if obj.shape[0] == len(omode_occu):
            print(f"obj.shape[0] is consistent with len(omode_occu) as '{obj.shape[0]}'")
        else:
            raise ValueError(f"Found inconsistency between obj.shape[0]({obj.shape[0]}) and len(omode_occu)({len(omode_occu)})")
        
        if obj.shape[1] == Nlayer:        
            print(f"obj.shape[1] is consistent with Nlayer as '{Nlayer}'")
        else:
            raise ValueError(f"Found inconsistency between obj.shape[1]({obj.shape[1]}) and Nlayer({Nlayer})")

        # Check obj tilts
        if obj_tilts is None:
            print(f"obj_tilts is None")
        else:
            if len(obj_tilts) in [1, N_scans]:
                print(f"obj_tilts is consistent with either 1 or N_scans")
            else:
                raise ValueError(f"Found inconsistency between len(obj_tilts) ({len(obj_tilts)}), 1, and N_scans({N_scans})")
        
        print(f"Pass the consistency chcek of initialized variables, initialization is done!")
    
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