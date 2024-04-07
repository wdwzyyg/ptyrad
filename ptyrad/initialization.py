## Initialization
from typing import Dict, List, Tuple

import numpy as np
from .data_io import load_fields_from_mat, load_hdf5, load_pt, load_tif
from .utils import kv2wavelength, near_field_evolution, hermite_like, make_stem_probe, make_mixed_probe, get_default_probe_simu_params


class Initializer:
    def __init__(self, exp_params, source_params):
        self.init_params = {'exp_params':exp_params, 'source_params':source_params}
        self.init_variables = {}
    
    def init_exp_params(self):
        print("\n### Initializing exp_params ###")
        exp_params = self.init_params['exp_params']       
        for key, value in exp_params.items():
            print(f"{key}: {value}")        
            
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
        
        # Correct negative values if any
        if (cbeds < 0).any():
            print(f"Imported meausrements int. statistics (min, mean, max) = ({cbeds.min():.4f}, {cbeds.mean():.4f}, {cbeds.max():.4f})")
            min_value = cbeds.min()
            cbeds -= min_value
            print(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements")
        
        # Permute, reshape, and flip
        if self.init_params['exp_params']['cbeds_permute'] is not None:
            permute_order = self.init_params['exp_params']['cbeds_permute']
            print("Permuting measurements")
            cbeds = cbeds.transpose(permute_order)
            
        if self.init_params['exp_params']['cbeds_reshape'] is not None:
            cbeds_shape = self.init_params['exp_params']['cbeds_reshape']
            print("Reshaping measurements")
            cbeds = cbeds.reshape(cbeds_shape)
            
        if self.init_params['exp_params']['cbeds_flip'] is not None:
            flip_axes = self.init_params['exp_params']['cbeds_flip']
            print("Flipping measurements")
            cbeds = np.flip(cbeds, flip_axes)
            
        # Normalizing cbeds
        print("Normalizing measurements so the averaged measurement has max intensity at 1")
        cbeds = cbeds / (np.mean(cbeds, 0).max()) # Normalizing the cbeds_data so that the averaged CBED has max at 1. This will make each CBED has max somewhere ~ 1
        cbeds = cbeds.astype('float32')
        # Print out some measurements statistics
        print(f"meausrements int. statistics (min, mean, max) = ({cbeds.min():.4f}, {cbeds.mean():.4f}, {cbeds.max():.4f})")
        print(f"measurements                      (N, Ky, Kx) = {cbeds.dtype}, {cbeds.shape}")
        self.init_variables['measurements'] = cbeds
        
    def init_obj(self):
        source = self.init_params['source_params']['obj_source']
        params = self.init_params['source_params']['obj_params']
        print(f"\n### Initializing obj from '{source}' ###")
        
        # Load file
        if source   == 'custom':
            obj = params
        elif source == 'pt':
            pt_path = params
            ckpt = load_pt(pt_path)
            obja, objp = ckpt['nn_params.obja'].detach().cpu().numpy(), ckpt['nn_params.objp'].detach().cpu().numpy()
            obj = obja * np.exp(1j * objp)
        elif source == 'PtyShv':
            mat_path = params
            obj = load_fields_from_mat(mat_path, 'object')[0]
            print("Expanding PtyShv object dimension")
            print(f"Input PtyShv obj has original shape {obj.shape}")
            if len(obj.shape) == 2: # Single-slice ptycho
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3: # MS-ptycho
                obj = obj[None,].transpose(0,3,1,2)
        elif source == 'simu':
            obj_shape = params
            if len(obj_shape) != 4:
                raise ValueError(f"Input `obj_shape` = {obj_shape}, please provide a total dimension of 4 as (omode, Nz, Ny, Nx) instead!")
            else:
                obj = np.exp(1j * 1e-8*np.random.rand(*obj_shape))
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        
        omode_max = self.init_params['exp_params']['omode_max']
        obj = obj[:omode_max].astype('complex64')
        print(f"object                    (omode, Nz, Ny, Nx) = {obj.dtype}, {obj.shape}")
        self.init_variables['obj'] = obj
            
    def init_probe(self):
        source = self.init_params['source_params']['probe_source']
        params = self.init_params['source_params']['probe_params']
        print(f"\n### Initializing probe from '{source}' ###")

        # Load file
        if source   == 'custom':
            probe = params
        elif source == 'pt':
            pt_path = params
            ckpt = load_pt(pt_path)
            probe = ckpt['nn_params.probe'].detach().cpu().numpy()
        elif source == 'PtyShv':
            mat_path = params
            probe = load_fields_from_mat(mat_path, 'probe')[0] # PtychoShelves probe generally has (Ny,Nx,pmode,vp) dimension. Usually people prefer pmode over vp.
            print(f"Input PtyShv probe has original shape {probe.shape}")
            if probe.ndim == 4:
                print(f"Import only the 1st variable probe mode") # I don't find variable probe modes are particularly useful for electon ptychography
                probe = probe[...,0]
            elif probe.ndim == 2:
                print("Expanding PtyShv probe dimension")
                probe = probe[...,None]
            else:
                probe = probe
            print("Permuting PtyShv probe") # For PtychoShelves input, do the transpose
            probe = probe.transpose(2,0,1)
        elif source == 'simu':
            probe_simu_params = params
            if probe_simu_params is None or type(probe_simu_params) == str:
                print(f"exp_params[`probe_simu_params`] is incorrecly set to `{probe_simu_params}`, use exp_params and default values instead for simulation")
                probe_simu_params = get_default_probe_simu_params(self.init_params['exp_params'] )
            probe = make_stem_probe(probe_simu_params)
            probe = make_mixed_probe(probe, probe_simu_params['pmodes'], probe_simu_params['pmode_init_pows'])
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        
        # Postprocess
        if self.init_params['exp_params']['probe_permute'] is not None and source != 'PtyShv':
            permute_order = self.init_params['exp_params']['probe_permute']
            print("Permuting probe")
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
        probe = probe[:pmode_max]
        probe = probe / (np.sum(np.abs(probe)**2)/np.sum(cbeds)*len(cbeds))**0.5 # Normalizing the probe_data so that the sum(|probe_data|**2) is the same with an averaged single CBED
        probe = probe.astype('complex64')
        print(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}")
        print(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.02f}, while sum(cbeds)/len(cbeds) = {np.sum(cbeds)/len(cbeds):.02f}")
        self.init_variables['probe'] = probe
            
    def init_pos(self):
        source = self.init_params['source_params']['pos_source']
        params = self.init_params['source_params']['pos_params']
        print(f"\n### Initializing probe pos from '{source}' ###")

        # Load file
        if source   == 'custom':
            crop_pos = params[0]
            probe_pos_shifts = params[1]
        elif source == 'pt':
            ckpt = load_pt(params)
            crop_pos         = ckpt['model_crop_pos'].detach().cpu().numpy()
            probe_pos_shifts = ckpt['nn_params.probe_pos_shifts'].detach().cpu().numpy()
        elif source == 'PtyShv':
            mat_contents = load_fields_from_mat(params, ['outputs.probe_positions', 'probe', 'object'])
            probe_positions = mat_contents[0]
            probe_shape = mat_contents[1].shape   # Matlab probe is (Ny,Nx,pmode,vp)
            obj_shape   = mat_contents[2].shape   # Matlab object is (Ny, Nx, Nz) or (Ny,Nx)
            pos_offset = np.ceil((np.array([obj_shape[0], obj_shape[1]])/2) - (np.array([probe_shape[0], probe_shape[1]])/2)) - 1 # For Matlab - Python index shift
            probe_positions_yx   = probe_positions[:, [1,0]] # The first index after shifting is the row index (along vertical axis)
            crop_coordinates     = probe_positions_yx + pos_offset 
            crop_pos = np.round(crop_coordinates) # This one is rounded
            probe_pos_shifts = (crop_coordinates - np.round(crop_coordinates)) # This shift (tH, tW) would be added to the probe to compensate the integer obj cropping
        elif source == 'simu':
            crop_pos = None
            probe_pos_shifts = None
        else:
            raise KeyError(f"File type {source} not implemented yet, please use 'custom', 'pt', 'PtyShv', or 'simu'!")
        crop_pos = crop_pos.astype('int16')
        probe_pos_shifts = probe_pos_shifts.astype('float32')
        print(f"crop_pos                                (N,2) = {crop_pos.dtype}, {crop_pos.shape}")
        print(f"probe_pos_shifts                        (N,2) = {probe_pos_shifts.dtype}, {probe_pos_shifts.shape}")
        self.init_variables['crop_pos'] = crop_pos
        self.init_variables['probe_pos_shifts'] = probe_pos_shifts
                
    def init_omode_occu(self):
        source = self.init_params['source_params']['omode_occu_source']
        params = self.init_params['source_params']['omode_occu_params']
        print(f"\n### Initializing omode_occu from '{source}' ###")

        if source   == 'custom':
            omode_occu = params
        elif source == 'uniform':
            omode = self.init_params['exp_params']['omode_max']
            omode_occu = np.ones(omode)/omode
        else:
            raise KeyError(f"Initialization method {source} not implemented yet, please use 'custom' or 'uniform'!")
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
        _, H, _, _ = near_field_evolution(probe_shape, z_distance, lambd, extent)
        H = H.astype('complex64')
        print(f"H                                    (Ky, Kx) = {H.dtype}, {H.shape}")
        self.init_variables['H'] = H
    
    def init_all(self):
        # Run this method to initialize all
        
        self.init_exp_params()
        self.init_measurements()
        self.init_obj()
        self.init_probe()
        self.init_pos()
        self.init_omode_occu()
        self.init_H()
        
        return self



