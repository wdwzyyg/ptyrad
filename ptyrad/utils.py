import os
os.environ["OMP_NUM_THREADS"] = "4" # This suppress the MiniBatchKMeans Windows MKL memory leak warning from make_batches
from random import shuffle
import torch
import warnings
import numpy as np
from time import time
from torch.fft import fft2, ifft2, ifftshift, fftshift
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

def get_size_bytes(x):
    
    print(f"Input tensor has shape {x.shape}, dtype {x.dtype}, and live on {x.device}")
    size_bytes = torch.numel(x) * x.element_size()
    size_mib = size_bytes / (1024 * 1024)
    size_gib = size_bytes / (1024 * 1024 * 1024)
    
    if size_bytes < 128 * 1024 * 1024:
        print(f"The size of the tensor is {size_mib:.2f} MiB")
    else:
        print(f"The size of the tensor is {size_gib:.2f} GiB")
    return size_bytes

def time_sync():
    torch.cuda.synchronize()
    t = time()
    return t

def select_center_rectangle_indices(matrix_height, matrix_width, height_rec, width_rec):
    ''' Select the indices from the center part of the 4D-STEM data '''
    # Thie is useful if you only want to reconstruct a small part of the data
    # Example
    # #indices = np.array(select_center_rectangle_indices(87,82,32,32))
    
    # Calculate the coordinates of the center rectangle
    start_row = (matrix_height - height_rec) // 2
    end_row = start_row + height_rec
    start_col = (matrix_width - width_rec) // 2
    end_col = start_col + width_rec

    # Generate flattened indices for the center rectangle
    indices = []
    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            indices.append(row * matrix_width + col)

    return indices

def make_save_dict(model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, iter, batch_losses):
    ''' Make a dict to save relevant paramerers '''
    
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    
    save_dict = {
                'optimizable_tensors':model.optimizable_tensors,
                'exp_params':exp_params,
                'source_params':source_params,
                'loss_params':loss_params,
                'constraint_params':constraint_params,
                'model_params':
                    {'lr_params':model.lr_params,
                    'omode_occu':model.omode_occu,
                    'H':model.H,
                    'crop_pos':model.crop_pos,
                    'shift_probes':model.shift_probes},
                'recon_params':recon_params,
                'loss_iters': loss_iters,
                'iter_t': iter_t,
                'iter': iter,
                'avg_losses': avg_losses
                }
    
    return save_dict

def make_recon_params_dict(NITER, BATCH_SIZE, GROUP, batches, output_path):
    recon_params = {
        'NITER':       NITER,
        'BATCH_SIZE':  BATCH_SIZE,
        'GROUP':       GROUP,
        'batches':     batches,
        'output_path': output_path
    }
    return recon_params

def make_batches(indices, pos, batch_size, group='random'):
    ''' Make batches from input indices '''
    # Input:
    #   indices: int, (Ns,) array. indices could be a subset of all indices.
    #   pos: int/float (N,2) array. Always pass in the full positions.
    #   batch_size: int. The number of indices of each mini-batch
    #   group: str. Choose between 'random', 'compact', or 'sparse' grouping.
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    #   The actual batch size would only be "close" if it's not divisible by len(indices) for 'random' grouping
    #   For 'compact' or 'sparse', it's generally fluctuating around the specified batch size

    num_batch = len(indices) // batch_size   

    if group == 'random':
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
        random_batches = np.array_split(shuffled_indices, num_batch) 
        return random_batches
        
    else:
        # Choose the selected pos from indices
        pos_s = pos[indices]
        # Kmeans for clustering
        kmeans = MiniBatchKMeans(init="k-means++", n_init=10, n_clusters=num_batch, max_iter=10, batch_size=3072)
        kmeans.fit(pos_s)
        labels = kmeans.labels_
        
        # Separate data points into groups
        compact_batches = []
        for batch_idx in range(num_batch):
            batch_indices_s = np.where(labels == batch_idx)[0]
            compact_batches.append(indices[batch_indices_s])

        if group == 'compact':
            return compact_batches

        else:
            # Initialize the list to store groups
            sparse_batches = [[] for _ in range(num_batch)]
            # Calculate the centroid for each compact group as initial start for sparse groups
            centroids = np.array([np.mean(pos[cbatch], axis=0) for cbatch in compact_batches])
            pairwise_distances = cdist(pos, pos) # Calculae the dist for all pos can keep the absolute index and skip the conversion
            
            # Start with the first point as a seed for each group
            for batch_idx in range(num_batch):
                distances = np.linalg.norm(pos_s - centroids[batch_idx], axis=1)
                closest_idx_s = np.argmin(distances)
                closest_idx = indices[closest_idx_s]
                sparse_batches[batch_idx].append(closest_idx)

            # Iterate through points
            for i in range(num_batch, len(pos_s)):
                min_distances = []
                # Iterate through groups
                for batch_idx in range(num_batch):
                    distances = pairwise_distances[sparse_batches[batch_idx], indices[i]]
                    min_distances.append(np.min(distances))
                
                max_group_index = np.argmax(min_distances)

                # Add the point to the group with the farthest minimal distance
                sparse_batches[max_group_index].append(indices[i])
            
            return sparse_batches

def shuffle_batches(batches, batch_size, group):
    ''' Shuffle the sequence of batches '''
    # Note: This will only shuffle the sequence of batches for "sparse" and "compact"
    # For random grouping, generate an entirely new random batches should be more appropriate for the purpose of randomness

    if group =='random':
        indices = np.concatenate(batches)        
        num_batch = len(indices) // batch_size   
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
        shuffled_batches = np.array_split(shuffled_indices, num_batch)
        return shuffled_batches 
    else:
        shuffled_batches = batches.copy()
        # In-place shuffling
        shuffle(shuffled_batches)
    
    return shuffled_batches

def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images with support for a batch of subpixel shifts.
    
    This function shifts a complex-valued input image by applying phase shifts in the Fourier domain,
    accommodating subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. It should be a complex-valued tensor with shape=(C, Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively.

    Outputs:
        shifted_img (torch.Tensor): The batch of shifted images. It has the same shape as the input batch of images, i.e., shape=(Nb, C, Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are specified as fractions of a pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - For more general usage of multidimensional array with lots of leading dimensions, we could modify the number of singletons in shifts and grid.
    """
    
    # img = (C,Ny,Nx), currently expecting mixed-state complex probe as a (pmode, Ny, Nx) complex64 tensor.
    # shifts = (Nb, 2)
    # grid = (2, Ny, Nx)
    
    shift_y, shift_x = shifts[:, 0, None, None, None], shifts[:, 1, None, None, None] # shift_y = (Nb, 1, 1, 1)
    ky, kx = grid[None, None, 0], grid[None,None, 1]                                  # ky = (1, 1, Ny, Nx)
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx) + shift_y * ky) 
    shifted_img = ifft2(ifftshift(fftshift(fft2(img[None,...])) * w))
    
    return shifted_img

def near_field_evolution(u_0_shape, z, lambd, extent, use_ASM_only=True, use_np_or_cp='np'):
#  FUNCTION  [u_1, H, h, dH] = near_field_evolution(u_0, z, lambda, extent, use_ASM_only)
#  Description: nearfield evolution function, it automatically switch
#  between ASM and Fraunhofer propagation 
#  Translated from Yi's fold_slice Matlab implementation into CuPy and NumPy by Chia-Hao Lee
    
    if use_np_or_cp == 'cp':
        import cupy as xp
    else:
        import numpy as xp

    u_0 = xp.ones(u_0_shape)
    
    H = None
    h = None
    u_1 = None
    dH = None

    if z == 0:
        H = 1
        u_1 = u_0
        return u_1, H, h, dH

    if z == float('inf'):
        return u_1, H, h, dH

    Npix = u_0.shape

    xgrid = xp.linspace(0.5 + (-Npix[0] / 2), 0.5 + (Npix[0] / 2 - 1), Npix[0]) / Npix[0]

    ygrid = xp.linspace(0.5 + (-Npix[1] / 2), 0.5 + (Npix[1] / 2 - 1), Npix[1]) / Npix[1]


    k = 2 * xp.pi / lambd

    extent = xp.array(extent)
    lambd = xp.array(lambd)
    z = xp.array(z)
    Npix = xp.array(Npix)

    F = xp.mean(extent ** 2 / (lambd * z * Npix))
    
    if abs(F) < 1 and not use_ASM_only:
        # Farfield propagation
        print('Farfield regime, F/Npix={:.2f}'.format(float(F)))
        Xrange = xgrid * extent[0]
        Yrange = ygrid * extent[1]
        X, Y = xp.meshgrid(Xrange, Yrange)
        h = xp.exp(1j * k * z + 1j * k / (2 * z) * (X.T ** 2 + Y.T ** 2))
        H = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(h)))
        H = H / xp.abs(H[Npix[0] // 2, Npix[1] // 2])  # Renormalize to conserve flux in the image
    else:
        # Standard ASM
        kx = 2 * xp.pi * xgrid / extent[0] * Npix[0]
        ky = 2 * xp.pi * ygrid / extent[1] * Npix[1]
        Kx, Ky = xp.meshgrid(kx, ky)
        dH = -1j * (Kx.T ** 2 + Ky.T ** 2) / (2 * k)
        H = xp.exp(1j * z * xp.sqrt(k ** 2 - Kx.T ** 2 - Ky.T ** 2))
        h = None

    # Do the ifftshift inside the function so the output has zero frequency at the center
    H = xp.fft.ifftshift(H)
    return u_1, H, h, dH

def test_loss_fn(model, indices, loss_fn):
    """ Print loss values for each term for convenient weight tuning """
    # model: PtychoAD model
    # indices: array-like indices indicating which probe position to evaluate
    # measurements: 4D-STEM data that's already passed to DEVICE
    # loss_fn: loss function object created from CombinedLoss
    
    with torch.no_grad():
        model_CBEDs, objp_patches = model(indices)
        measured_CBEDs = model.get_measurements(indices)
        _, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)

        # Print loss_name and loss_value with padding
        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            print(f"{loss_name.ljust(11)}: {loss_value.detach().cpu().numpy():.8f}")
    return

def kv2wavelength(acceleration_voltage):
    # Physical Constants
    PLANCKS = 6.62607015E-34 # m^2*kg / s
    REST_MASS_E = 9.1093837015E-31 # kg
    CHARGE_E = 1.602176634E-19 # coulomb 
    SPEED_OF_LIGHT = 299792458 # m/s

    # Useful constants in EM unit 
    hc = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
    REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2
    
    wavelength = hc/np.sqrt((2*REST_ENERGY_E + acceleration_voltage)*acceleration_voltage) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))

    return wavelength

def get_default_probe_simu_params(exp_params):
    probe_simu_params = { ## Basic params
                    "kv"             : exp_params['kv'],
                    "conv_angle"     : exp_params['conv_angle'],
                    "Npix"           : exp_params['Npix'],
                    "rbf"            : exp_params['rbf'], # dk = conv_angle/1e3/rbf/wavelength
                    "dx"             : exp_params['dx_spec'], # dx = 1/(dk*Npix) #angstrom
                    "print_info"     : False,
                    "pmodes"         : exp_params['pmode_max'],
                    "pmode_init_pows": exp_params['pmode_init_pows'],
                    ## Aberration coefficients
                    "df": exp_params['defocus'], #first-order aberration (defocus) in angstrom, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland's notation
                    "c3": exp_params['c3'] , #third-order spherical aberration in angstrom
                    "c5":0, #fifth-order spherical aberration in angstrom
                    "c7":0, #seventh-order spherical aberration in angstrom
                    "f_a2":0, #twofold astigmatism in angstrom
                    "f_a3":0, #threefold astigmatism in angstrom
                    "f_c3":0, #coma in angstrom
                    "theta_a2":0, #azimuthal orientation in radian
                    "theta_a3":0, #azimuthal orientation in radian
                    "theta_c3":0, #azimuthal orientation in radian
                    "shifts":[0,0], #shift probe center in angstrom
                    }
    return probe_simu_params

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
    
    print("Start simulating STEM probe")
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
        print(f'alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
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
    print(f"Start making mixed-state STEM probe with {pmodes} incoherent probe modes")
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

def check_modes_ortho(tensor, atol = 2e-5):
    ''' Check if the modes in tensor (Nmodes, []) is orthogonal to each other'''
    # The easiest way to check orthogonality is to calculate the dot product of their 1D vector views
    # Orthogonal vectors would have dot product equals to 0 (Note that `orthonormal` also requires they have unit length)
    # Note that due to the floating point precision, we should set a reasonable tolerance w.r.t 0.
    
    print(f"Input tensor has shape {tensor.shape} and dtype {tensor.dtype}")
    for i in range(tensor.shape[0]):
        for j in range(i + 1, tensor.shape[0]):
            dot_product = torch.dot(tensor[i].view(-1), tensor[j].view(-1))
            if torch.allclose(dot_product, torch.tensor(0., dtype=dot_product.dtype, device=dot_product.device), atol=atol):
                print(f"Modes {i} and {j} are orthogonal with abs(dot) = {dot_product.abs().detach().cpu().numpy()}")
            else:
                print(f"Modes {i} and {j} are not orthogonal with abs(dot) = {dot_product.abs().detach().cpu().numpy()}")

###################################### ARCHIVE ##################################################

def make_compact_batches(pos, num_groups):
    from sklearn.cluster import MiniBatchKMeans
    # Fit K-Means clustering algorithm
    kmeans = MiniBatchKMeans(init="k-means++", n_init=10, n_clusters=num_groups, max_iter=10, batch_size=3072)
    kmeans.fit(pos)

    # Get cluster labels
    labels = kmeans.labels_

    # Separate data points into groups
    groups = []
    for i in range(num_groups):
        group_indices = np.where(labels == i)[0]
        groups.append(group_indices)

    return groups

def make_random_batches(indices, batch_size):
    ''' Make random batches from input indices and batch_size '''
    # Input:
    #   indices: int or array_like. If indices is an integer, randomly permute np.arange(indices). If indices is an array, make a copy and shuffle the elements randomly.
    #   batch_size: int. The number of indices of each mini-batch
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    # The actual batch size would only be "close" if it's not divisible by len(indices)
    if isinstance(indices, int):
        indices = range(indices)
    num_batch = len(indices) / batch_size                 
    rng = np.random.default_rng()
    shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
    batches = np.array_split(shuffled_indices, num_batch) 
    return batches

def cplx_from_np(a, cplx_type='amp_phase', ndim = -1):
    """ Transform a complex numpy array in a "pseudo-complex" tensor"""
    # a: Input complex np array
    # ndim: The axis that stacks the real/imag or amp/phase part
    # cplx_type: "real_imag" or "amp_phase"
    # return: pseuso-complex array shaped (...,2)
    
    if cplx_type == "real_imag":
        return torch.stack([torch.from_numpy(a).real, torch.from_numpy(a).imag], ndim)
    elif cplx_type == "amp_phase":
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)
    else:
        warnings.warn("cplx_type {} not implemented. Defaulting to 'amp_phase'.".format(cplx_type))
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)

def complex_object_interp3d(complex_object, zoom_factors, z_axis, use_np_or_cp='np'):
    """
    Interpolate a 3D complex object while preserving multiscattering behavior.

    Parameters:
    - complex_object (ndarray): Input complex object with shape (z, y, x).
    - zoom_factors (tuple): Tuple of zoom factors for (z, y, x).
    = z_axis: int indicating the z-axis posiiton
    - use_np_or_cp (str): Specify the library to use, 'np' for NumPy or 'cp' for CuPy.

    Returns:
    ndarray: Interpolated complex object with the same dtype as the input.

    Notes:
    - Amplitude and phase are treated separately as they obey different conservation laws.
    - Phase shift for multiple z-slices is additive, ensuring the sum of all z-slices remains the same.
    - Amplitude between each z-slice is multiplicative. Linear interpolation of log(amplitude) is performed
      while maintaining the conservation law.
    - The phase of the object should be unwrapped and smooth.
    - If possible, use cupy for 40x faster speed (I got 1 sec vs 40 sec for 320*320*420 target size in a one-shot calculation on my Quadro P5000)

    Example:
    >>> complex_object = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)
    >>> zoom_factors = (2, 2, 1.5)
    >>> result = complex_object_interp3d(complex_object, zoom_factors, use_np_or_cp='np')
    """
    
    if use_np_or_cp == 'cp':
        import cupy as xp
        from cupyx.scipy import ndimage
        complex_object = xp.array(complex_object)
    else:
        import numpy as xp
        from scipy import ndimage
    
    if zoom_factors == (1,1,1):
        print(f"No interpolation is needed, returning original object with shape = {complex_object.shape}.")
        return complex_object

    else:
        obj_dtype = complex_object.dtype
        obj_a = xp.abs(complex_object)
        obj_p = xp.angle(complex_object)
        
        obj_a_interp = xp.exp(ndimage.zoom(xp.log(obj_a), zoom_factors) / zoom_factors[z_axis])
        obj_p_interp = ndimage.zoom(obj_p, zoom_factors) / zoom_factors[z_axis]
        
        complex_object_interp3d = obj_a_interp * xp.exp(obj_p_interp*1j)
        print(f"The object shape is interpolated to {complex_object_interp3d.shape}.")
        return complex_object_interp3d.astype(obj_dtype)

def imshift(img, shifts, device):
    """
    Generates a shifted image with support for subpixel shifts.
    
    This function shifts a complex-valued input image by applying phase shifts in the Fourier domain,
    accommodating subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. It should be a complex-valued tensor with shape=(..., Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a single tensor as (shift_y, shift_x).
        device (torch.device): The device on which the computation will be performed.

    Outputs:
        shifted_img (torch.Tensor): The shifted image. It has the same shape as the input image, i.e., shape=(..., Ny, Nx).

    Note:
        - The shifts are specified as fractions of a pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
    """

    Ny, Nx = img.shape[-2], img.shape[-1]
    shift_y, shift_x = shifts[0], shifts[1]
    ry, rx = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing='ij')
    ry, rx = ry.to(device), rx.to(device)
    w = -torch.exp(-(2j * torch.pi) * (shift_x * rx / Nx  + shift_y * ry / Ny))
    shifted_img = ifft2(ifftshift(fftshift(fft2(img)) * w))

    return shifted_img

def Fresnel_propagator(probe, z_distances, lambd, extent):
    # Positive z_distance is adding more overfocus, or letting the probe to forward propagate more
    
    # Example usage
    # dfs = np.linspace(0,200,100)
    # prop_probes = Fresnel_propagator(probe_data, dfs, lambd, extent)
    # print(f"probe_data.shape = {probe_data.shape}, prop_probes.shape = {prop_probes.shape}")
    # print(f"sum(abs(probe)**2) = {np.sum(np.abs(probe_data)**2)}, \nsum(abs(prop_probes)**2) = {np.sum(np.abs(prop_probes)**2, axis=(-3,-2,-1))}")
    
    
    # dfs = [-3,-2,-1,0]
    # prop_probes = Fresnel_propagator(probe_data, dfs, lambd, extent)
    # print(f"probe_data.shape = {probe_data.shape}, prop_probes.shape = {prop_probes.shape}")
    # print(f"sum(abs(probe)**2) = {np.sum(np.abs(probe_data)**2)}, \nsum(abs(prop_probes)**2) = {np.sum(np.abs(prop_probes)**2, axis=(-3,-2,-1))}")

    # plt.figure()
    # plt.title("probe int x-z")
    # plt.imshow(np.abs(prop_probes[:,0,prop_probes.shape[-2]//2,:])**2, aspect=10)
    # plt.yticks(np.arange(0, prop_probes.shape[0]), dfs)
    # plt.ylabel('Ang along z')
    # plt.colorbar()
    # plt.show()
    
    prop_probes = np.zeros((len(z_distances), *probe.shape)).astype(probe.dtype)
    for i, z_distance in enumerate(z_distances):
        _, H, _, _ = near_field_evolution(probe.shape[-2:], z_distance, lambd, extent, use_ASM_only=True, use_np_or_cp='np')
        prop_probes[i] = np.fft.ifft2(H * np.fft.fft2(probe, axes=(-2, -1)), axes=(-2, -1))
    
    return prop_probes

