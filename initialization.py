## Initialization

import numpy as np
from data_io import load_fields_from_mat, load_hdf5, load_pt, load_tif
from utils import kv2wavelength, near_field_evolution

class Initializer:
    def __init__(self, init_params):
        self.init_params = init_params
        self.init_variables = {}
    
    def init_exp_params(self):
        print("\n### Initializing exp_params ###")
        exp_params = self.init_params['exp_params']       
        for key, value in exp_params.items():
            print(f"{key}: {value}")        
            
    def init_measurements(self):
        source = self.init_params['measurements']["source"]
        params = self.init_params['measurements']['params']
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
            print(f"File type {source} not implemented yet, didn't load anything!")
        
        # Postprocess
        if self.init_params['exp_params']['cbeds_permute'] is not None:
            permute_order = self.init_params['exp_params']['cbeds_permute']
            print("Permuting measurements")
            cbeds = cbeds.transpose(permute_order)
            
        if self.init_params['exp_params']['cbeds_reshape'] is not None:
            cbeds_shape = self.init_params['exp_params']['cbeds_reshape']
            print("Reshaping measurements")
            cbeds = cbeds.reshape(cbeds_shape)
            
        # Normalizing cbeds
        cbeds = cbeds / (np.mean(cbeds, 0).max()) # Normalizing the cbeds_data so that the averaged CBED has max at 1. This will make each CBED has max somewhere ~ 1
        cbeds = cbeds.astype('float32')
        print(f"measurements                      (N, Ky, Kx) = {cbeds.dtype}, {cbeds.shape}")
        self.init_variables['measurements'] = cbeds
        
    def init_obj(self):
        source = self.init_params['obj']["source"]
        params = self.init_params['obj']['params']
        print(f"\n### Initializing obj from '{source}' ###")
        
        # Load file
        if source   == 'custom':
            obj = params
        elif source == 'pt':
            pt_path = params
            ckpt = load_pt(pt_path)
            obja, objp = ckpt['nn_params.obja'].detach().cpu().numpy(), ckpt['nn_params.objp'].detach().cpu().numpy()
            obj = obja * np.exp(1j * objp)
        elif source == 'mat':
            mat_path = params
            obj = load_fields_from_mat(mat_path, 'object')[0]
            print("Expanding .mat object dimension")
            if len(obj.shape) == 2:
                obj = obj[None,None,:,:]
            elif len(obj.shape)==3:
                obj = obj[None,].transpose(0,3,1,2)
        elif source == 'simu':
            obj_shape = params
            obj = np.exp(1j * 1e-8*np.random.rand(*obj_shape))
        else:
            print(f"File type {source} not implemented yet, didn't load anything!")
        
        omode_max = self.init_params['exp_params']['omode_max']
        obj = obj[:omode_max].astype('complex64')
        print(f"object                    (omode, Nz, Ny, Nx) = {obj.dtype}, {obj.shape}")
        self.init_variables['obj'] = obj
            
    def init_probe(self):
        source = self.init_params['probe']["source"]
        params = self.init_params['probe']['params']
        print(f"\n### Initializing probe from '{source}' ###")

        # Load file
        if source   == 'custom':
            probe = params
        elif source == 'pt':
            pt_path = params
            ckpt = load_pt(pt_path)
            probe = ckpt['nn_params.probe'].detach().cpu().numpy()
        elif source == 'mat':
            mat_path = params
            probe = load_fields_from_mat(mat_path, 'probe')[0]
        elif source == 'simu':
            probe_simu_params = params
            probe = make_stem_probe(probe_simu_params)
            probe = make_mixed_probe(probe, probe_simu_params['pmodes'], probe_simu_params['pmode_init_pows'])
        else:
            print(f"File type {source} not implemented yet, didn't load anything!")
        
        # Postprocess
        if self.init_params['exp_params']['probe_permute'] is not None:
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
        cbeds = self.init_variables['measurements']
        probe = probe[:pmode_max]
        probe = probe / (np.sum(np.abs(probe)**2)/np.sum(cbeds)*len(cbeds))**0.5 # Normalizing the probe_data so that the sum(|probe_data|**2) is the same with an averaged single CBED
        probe = probe.astype('complex64')
        print(f"probe                         (pmode, Ny, Nx) = {probe.dtype}, {probe.shape}")
        print(f"sum(|probe_data|**2) = {np.sum(np.abs(probe)**2):.02f}, while sum(cbeds)/len(cbeds) = {np.sum(cbeds)/len(cbeds):.02f}")
        self.init_variables['probe'] = probe
            
    def init_pos(self):
        source = self.init_params['pos']["source"]
        params = self.init_params['pos']['params']
        print(f"\n### Initializing probe pos from '{source}' ###")

        # Load file
        if source   == 'custom':
            crop_pos = params[0]
            probe_pos_shifts = params[1]
        elif source == 'pt':
            ckpt = load_pt(params)
            crop_pos         = ckpt['model_crop_pos'].detach().cpu().numpy()
            probe_pos_shifts = ckpt['nn_params.probe_pos_shifts'].detach().cpu().numpy()
        elif source == 'mat':
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
            print(f"File type {source} not implemented yet, didn't load anything!")
        crop_pos = crop_pos.astype('int16')
        probe_pos_shifts = probe_pos_shifts.astype('float32')
        print(f"crop_pos                                (N,2) = {crop_pos.dtype}, {crop_pos.shape}")
        print(f"probe_pos_shifts                        (N,2) = {probe_pos_shifts.dtype}, {probe_pos_shifts.shape}")
        self.init_variables['crop_pos'] = crop_pos
        self.init_variables['probe_pos_shifts'] = probe_pos_shifts
                
    def init_omode_occu(self):
        source = self.init_params['omode_occu']["source"]
        params = self.init_params['omode_occu']['params']
        print(f"\n### Initializing omode_occu from '{source}' ###")

        if source   == 'custom':
            omode_occu = params
        elif source == 'uniform':
            omode = len(self.init_variables['obj'])
            omode_occu = np.ones(omode)/omode
        else:
            print(f"Initialization method {source} not implemented yet, didn't initialize omode_occu!")
        omode_occu = omode_occu.astype('float32')
        print(f"omode_occu                            (omode) = {omode_occu.dtype}, {omode_occu.shape}")
        self.init_variables['omode_occu'] = omode_occu
        
    def init_H(self):
        print("\n### Initializing H (Fresnel propagator) ###")
        probe_shape = np.array(self.init_variables['probe'].shape[-2:])
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


def make_stem_probe(params_dict):
    # MAKE_TEM_PROBE Generate probe functions produced by object lens in 
    # transmission electron microscope.
    # Written by Yi Jiang based on Eq.(2.10) in Advanced Computing in Electron 
    # Microscopy (2nd edition) by Dr.Kirkland
    # Implemented and slightly modified in python by Chia-Hao Lee
 
    # Outputs:
        #  probe: complex probe functions at real space (sample plane)
    # Inputs: 
        #  params_dict: probe parameters and other settings
        
    ## Basic params
    voltage     = params_dict["kv"]         # Ang
    conv_angle  = params_dict["conv_angle"] # mrad
    Npix        = params_dict["Npix"]       # Number of pixel of thr detector/probe
    rbf         = params_dict["rbf"]        # Pixels of radius of BF disk, used to calculate dk
    dx          = params_dict["dx"]         # px size in Angstrom
    ## Aberration coefficients
    df          = params_dict["df"] #first-order aberration (defocus) in angstrom
    c3          = params_dict["c3"] #third-order spherical aberration in angstrom
    c5          = params_dict["c5"] #fifth-order spherical aberration in angstrom
    c7          = params_dict["c7"] #seventh-order spherical aberration in angstrom
    f_a2        = params_dict["f_a2"] #twofold astigmatism in angstrom
    f_a3        = params_dict["f_a3"] #threefold astigmatism in angstrom
    f_c3        = params_dict["f_c3"] #coma in angstrom
    theta_a2    = params_dict["theta_a2"] #azimuthal orientation in radian
    theta_a3    = params_dict["theta_a3"] #azimuthal orientation in radian
    theta_c3    = params_dict["theta_c3"] #azimuthal orientation in radian
    shifts      = params_dict["shifts"] #shift probe center in angstrom
    
    # Calculate some variables
    wavelength = 12.398/np.sqrt((2*511.0+voltage)*voltage) #angstrom
    k_cutoff = conv_angle/1e3/wavelength
    
    if rbf is not None and dx is None:
        print("Using 'rbf' for dk sampling")
        dk = conv_angle/1e3/wavelength/rbf
        dx = 1/(dk*Npix) # Populate dx with the calculated value
    elif dx is not None:
        print("Using 'dx' for dk sampling")
        dk = 1/(dx*Npix)
    else:
        raise ValueError("Either 'rbf' or 'dx' must be provided to calculate dk sampling.")
    
    # Make k space sampling and probe forming aperture
    kx = np.linspace(-np.floor(Npix/2),np.ceil(Npix/2)-1,Npix)
    [kX,kY] = np.meshgrid(kx,kx)

    kX = kX*dk
    kY = kY*dk
    kR = np.sqrt(kX**2+kY**2)
    theta = np.arctan2(kY,kX)
    mask = (kR<=k_cutoff).astype('bool') 
    
    # Adding aberration one-by-one, the aberrations modify the flat phase (imagine a flat wavefront at aperture plane) with some polynomial perturbations
    # The aberrated phase is called chi(k), probe forming aperture is placed here to select the relatively flat phase region to form desired real space probe
    # Note that chi(k) is real-valued function with unit as radian, it's also not limited between -pi,pi. Think of phase shift as time delay might help.
    
    chi = -np.pi*wavelength*kR**2*df
    if c3!=0: 
        chi += np.pi/2*c3*wavelength**3*kR**4
    if c5!=0: 
        chi += np.pi/3*c5*wavelength**5*kR**6
    if c7!=0: 
        chi += np.pi/4*c7*wavelength**7*kR**8
    if f_a2!=0: 
        chi += np.pi*f_a2*wavelength*kR**2*np.sin(2*(theta-theta_a2))
    if f_a3!=0: 
        chi += 2*np.pi/3*f_a3*wavelength**2*kR**3*np.sin(3*(theta-theta_a3))
    if f_c3!=0: 
        chi += 2*np.pi/3*f_c3*wavelength**2*kR**3*np.sin(theta-theta_c3)

    psi = np.exp(-1j*chi)*np.exp(-2*np.pi*1j*shifts[0]*kX)*np.exp(-2*np.pi*1j*shifts[1]*kY)
    probe = mask*psi # It's now the masked wave function at the aperture plane
    probe = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(probe))) # Propagate the wave function from aperture to the sample plane. 
    probe = probe/np.sqrt(np.sum((np.abs(probe))**2)) # Normalize the probe so sum(abs(probe)^2) = 1

    if params_dict['print_info']:
        # Print some useful values
        print(f'kv          = {voltage} kV')    
        print(f'wavelength  = {wavelength:.4f} Ang')
        print(f'conv_angle  = {conv_angle} mrad')
        print(f'Npix        = {Npix} px')
        print(f'dk          = {dk:.4f} Ang^-1')
        print(f'kMax        = {(Npix*dk/2):.4f} Ang^-1')
        print(f'dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
        print(f'Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
        print(f'Real space extent = {dx*Npix:.4f} Ang')
    
    return probe

def make_mixed_probe(probe, pmodes, pmode_init_pows):
    ''' Make a mixed state probe from a single state probe '''
    # Input:
    #   probe: (Ny,Nx) complex array
    #   pmodes: number of incoherent probe modes, scaler int
    #   pmode_init_pows: Integrated intensity of modes. List of a value (e.g. [0.02]) or a couple values for the first few modes. sum(pmode_init_pows) must < 1. 
    # Output:
    #   mixed_probe: A mixed state probe with (pmode,Ny,Nx)
    
    # Prepare a mixed-state probe `mixed_probe`
    M = np.ceil(pmodes**0.5)-1
    N = np.ceil(pmodes/(M+1))-1
    mixed_probe = hermite_like(probe, M,N)[:pmodes]
    
    # Normalize each pmode
    pmode_pows = np.zeros(pmodes)
    for ii in range(1,pmodes):
        if ii<np.size(pmode_init_pows):
            pmode_pows[ii] = pmode_init_pows[ii-1]
        else:
            pmode_pows[ii] = pmode_init_pows[-1]
    if sum(pmode_pows)>1:
        raise ValueError('Modes total power exceeds 1, check pmode_init_pows')
    else:
        pmode_pows[0] = 1-sum(pmode_pows)

    mixed_probe = mixed_probe * np.sqrt(pmode_pows)[:,None,None]
    print(f"Relative power of probe modes = {pmode_pows}")
    return mixed_probe

def hermite_like(fundam, M, N):
    # %HERMITE_LIKE
    # % Receives a probe and maximum x and y order M N. Based on the given probe
    # % and multiplying by a Hermitian function new modes are computed. The modes
    # % are then orthonormalized.
    
    # Input:
    #   fundam: base function
    #   X,Y: centered meshgrid for the base function
    #   M,N: order of the hermite_list basis
    # Output:
    #   H: 
    # Note:
    #   This function is a python implementation of `ptycho\+core\hermite_like.m` from PtychoShelves with some modification
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The X, Y meshgrid are moved into the funciton
    #   The H is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   Note that H would output (M+1)*(N+1) modes, which could be a bit more than the specified pmode
    
    
    # Initialize i/o
    M = M.astype('int')
    N = N.astype('int')
    m = np.arange(M+1)
    n = np.arange(N+1)
    H = np.zeros(((M+1)*(N+1), fundam.shape[-2], fundam.shape[-1]), dtype=fundam.dtype)
      
    # Create meshgrid
    rows, cols = fundam.shape[-2:]
    x = np.arange(cols) - cols / 2
    y = np.arange(rows) - rows / 2
    X, Y = np.meshgrid(x, y)
    
    cenx = np.sum(X * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    ceny = np.sum(Y * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    varx = np.sum((X - cenx)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    vary = np.sum((Y - ceny)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)

    counter = 0
    
    # Create basis
    for nii in n:
        for mii in m:
            auxfunc = ((X - cenx)**mii) * ((Y - ceny)**nii) * fundam
            if counter == 0:
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            else:
                auxfunc = auxfunc * np.exp(-((X - cenx)**2 / (2*varx)) - ((Y - ceny)**2 / (2*vary)))
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))

            # Now make it orthogonal to the previous ones
            for ii in range(counter): # The other ones
                auxfunc = auxfunc - np.dot(H[ii].reshape(-1), np.conj(auxfunc).reshape(-1)) * H[ii]

            # Normalize each mode so that their intensities sum to 1
            auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            H[counter] = auxfunc
            counter += 1

    return H