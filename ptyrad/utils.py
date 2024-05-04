import os
os.environ["OMP_NUM_THREADS"] = "4" # This suppress the MiniBatchKMeans Windows MKL memory leak warning from make_batches
import warnings
from time import time
from math import floor, ceil

from tifffile import imwrite
import numpy as np
import torch
from torch.fft import fft2, ifft2, ifftshift, fftshift
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

def has_nan_or_inf(tensor):
    """
    Check if a torch.Tensor contains any NaN or Inf values.

    Parameters:
        tensor (torch.Tensor): Input tensor to check.

    Returns:
        bool: True if the tensor contains any NaN or Inf values, False otherwise.
    """
    # Check for NaN values
    has_nan = torch.isnan(tensor).any()

    # Check for Inf values
    has_inf = torch.isinf(tensor).any()

    return has_nan or has_inf

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

def compose_affine_matrix(scale, asymmetry, rotation, shear):
    # Adapted from PtychoShelves +math/compose_affine_matrix.m
    # The input rotation and shear is in unit of degree
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    A1 = np.array([[scale, 0], [0, scale]])
    A2 = np.array([[1 + asymmetry/2, 0], [0, 1 - asymmetry/2]])
    A3 = np.array([[np.cos(rotation_rad), np.sin(rotation_rad)], [-np.sin(rotation_rad), np.cos(rotation_rad)]])
    A4 = np.array([[1, 0], [np.tan(shear_rad), 1]])
    
    affine_mat = A1 @ A2 @ A3 @ A4

    return affine_mat

def select_scan_indices(N_scan_slow, N_scan_fast, subscan_slow=None, subscan_fast=None, mode='full'):
    
    N_scans = N_scan_slow * N_scan_fast
    
    print(f"Selecting indices with the '{mode}' mode ")
    # Generate flattened indices for the entire FOV
    if mode == 'full':
        indices = np.arange(N_scans)
        return indices

    # Set default values for subscan params
    if subscan_slow is None and subscan_fast is None:
        print(f"Subscan params are not provided, setting subscans to default as half of the total scan for both directions")
        subscan_slow = N_scan_slow//2
        subscan_fast = N_scan_fast//2
        
    # Generate flattened indices for the center rectangular region
    if mode == 'center':
        print(f"Choosing subscan with {(subscan_slow, subscan_fast)}") 
        start_row = (N_scan_slow - subscan_slow) // 2
        end_row = start_row + subscan_slow
        start_col = (N_scan_fast - subscan_fast) // 2
        end_col = start_col + subscan_fast
        indices = np.array([row * N_scan_fast + col for row in range(start_row, end_row) for col in range(start_col, end_col)])

    # Generate flattened indices for the entire FOV with sub-sampled indices
    elif mode == 'sub':
        print(f"Choosing subscan with {(subscan_slow, subscan_fast)}") 
        full_indices = np.arange(N_scans).reshape(N_scan_slow, N_scan_fast)
        subscan_slow_id = np.linspace(0, N_scan_slow-1, num=subscan_slow, dtype=int)
        subscan_fast_id = np.linspace(0, N_scan_fast-1, num=subscan_fast, dtype=int)
        slow_grid, fast_grid = np.meshgrid(subscan_slow_id, subscan_fast_id, indexing='ij')
        indices = full_indices[slow_grid, fast_grid].reshape(-1)

    else:
        raise KeyError(f"Indices selection mode {mode} not implemented, please use either 'full', 'center', or 'sub'")   
        
    return indices

def make_save_dict(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses):
    ''' Make a dict to save relevant paramerers '''
    
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    
    save_dict = {
                'output_path'           : output_path,
                'optimizable_tensors'   : model.optimizable_tensors,
                'exp_params'            : exp_params,
                'source_params'         : source_params,
                'loss_params'           : loss_params,
                'constraint_params'     : constraint_params,
                'model_params':
                    {'recenter_cbeds'   : model.recenter_cbeds,
                     'detector_blur_std': model.detector_blur_std,
                     'lr_params'        : model.lr_params,
                     'omode_occu'       : model.omode_occu,
                     'H'                : model.H,
                     'z_distance'       : model.z_distance,
                     'crop_pos'         : model.crop_pos,
                     'shift_probes'     : model.shift_probes,
                     'avg_cbeds_shift'  : model.avg_cbeds_shift},
                'recon_params'          : recon_params,
                'loss_iters'            : loss_iters,
                'iter_t'                : iter_t,
                'niter'                 : niter,
                'avg_losses'            : avg_losses
                }
    
    return save_dict

def make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE, SAVE_ITERS):
    recon_params = {
        'NITER'         :        NITER,
        'INDICES_MODE'  : INDICES_MODE,
        'BATCH_SIZE'    :   BATCH_SIZE,
        'GROUP_MODE'    :   GROUP_MODE,
        'SAVE_ITERS'    :   SAVE_ITERS,
    }
    return recon_params

def make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, postfix=''):
    ''' Generate the output folder given indices, recon_params, model, and constraint_params '''
    
    # Note that if recon_params['SAVE_ITERS'] is None, the output_path is returned but not generated
    
    # # Example
    # NITER        = 50
    # INDICES_MODE = 'full' #'full', 'center', 'sub'
    # BATCH_SIZE   = 128
    # GROUP_MODE   = 'random' #'random', 'sparse', 'compact'

    # output_dir   = 'output/STO'
    # postfix      = ''

    # pos          = model.crop_pos.cpu().numpy()
    # indices      = select_scan_indices(exp_params['N_scan_slow'], exp_params['N_scan_slow'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE)
    # batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE)
    # recon_params = make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE)
    # output_path  = make_output_folder(output_dir, indices, recon_params, model, constraint_params, postfix)
    
    output_path  = output_dir
    cbeds_flipT  = exp_params['cbeds_flipT']
    indices_mode = recon_params['INDICES_MODE']
    group_mode   = recon_params['GROUP_MODE']
    batch_size   = recon_params['BATCH_SIZE']
    pmode        = model.opt_probe.size(0)
    dp_size      = model.measurements.size(1)
    obj_shape    = model.opt_objp.shape
    probe_lr     = format(model.lr_params['probe'], '.0e').replace("e-0", "e-") if model.lr_params['probe'] !=0 else 0
    objp_lr      = format(model.lr_params['objp'], '.0e').replace("e-0", "e-") if model.lr_params['objp'] !=0 else 0
    obja_lr      = format(model.lr_params['obja'], '.0e').replace("e-0", "e-") if model.lr_params['obja'] !=0 else 0
    pos_lr       = format(model.lr_params['probe_pos_shifts'], '.0e').replace("e-0", "e-") if model.lr_params['probe_pos_shifts'] !=0 else 0

    output_path  = output_dir + "/" + f"{indices_mode}_N{len(indices)}_dp{dp_size}"
    
    if cbeds_flipT is not None:
        output_path = output_path + '_flipT' + ''.join(str(x) for x in cbeds_flipT)
        
    output_path += f"_{group_mode}{batch_size}_p{pmode}_plr{probe_lr}_oalr{obja_lr}_oplr{objp_lr}_slr{pos_lr}_{obj_shape[0]}obj_{obj_shape[1]}slice"
    
    if obj_shape[1] != 1:
        z_distance = model.z_distance.cpu().numpy().round(2)
        output_path += f"_dz{z_distance}"
    
    if constraint_params['kz_filter']['freq'] is not None:
        obj_type = constraint_params['kz_filter']['obj_type']
        kz_str = {'both': 'kz', 'amplitude': 'kza', 'phase': 'kzp'}.get(obj_type)
        beta = constraint_params['kz_filter']['beta']
        output_path += f"_{kz_str}reg{beta}"
    
    if model.detector_blur_std is not None and model.detector_blur_std != 0:
        output_path += f"_dpblur{model.detector_blur_std}"
    
    if constraint_params['obj_blur']['freq'] is not None and constraint_params['obj_blur']['std'] != 0:
        obj_type = constraint_params['obj_blur']['obj_type']
        obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
        output_path += f"_{obj_str}blur{constraint_params['obj_blur']['std']}"
        
    if constraint_params['probe_mask_r']['freq'] is not None:
        output_path += f"_pmr{round(constraint_params['probe_mask_r']['radius'],2)}"
        
    if constraint_params['probe_mask_k']['freq'] is not None:
        output_path += f"_pmk{round(constraint_params['probe_mask_k']['radius'],2)}"
            
    output_path += postfix
    
    if recon_params['SAVE_ITERS'] is not None:
        os.makedirs(output_path, exist_ok=True)
        print(f"output_path = '{output_path}' is generated!")
    else:
        print(f"output_path = '{output_path}' but is NOT generated!")
    return output_path

def make_batches(indices, pos, batch_size, mode='random'):
    ''' Make batches from input indices '''
    # Input:
    #   indices: int, (Ns,) array. indices could be a subset of all indices.
    #   pos: int/float (N,2) array. Always pass in the full positions.
    #   batch_size: int. The number of indices of each mini-batch
    #   mode: str. Choose between 'random', 'compact', or 'sparse' grouping.
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    #   The actual batch size would only be "close" if it's not divisible by len(indices) for 'random' grouping
    #   For 'compact' or 'sparse', it's generally fluctuating around the specified batch size
    #   To check the correctness of each grouping, you may visualize the pos
    #   Also we want to make sure we're not missing any indices, so we can do:
    #
    #   flatten_indices = np.concatenate(batches)
    #   flatten_indices.sort()
    #   indices.sort()
    #   all(flatten_indices == indices)

    if len(indices) > len(pos):
        raise ValueError(f"len(indices) = '{len(indices)}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")
    
    if indices.max() > len(pos):
        raise ValueError(f"Maximum index '{indices.max()}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")

    num_batch = len(indices) // batch_size   
    t_start = time()
    if mode == 'random':
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
        random_batches = np.array_split(shuffled_indices, num_batch) 
        print(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec")
        return random_batches
        
    else: # Either 'compact' or 'sparse'
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

        if mode == 'compact':
            print(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec")
            return compact_batches

        else: # 'sparse' mode
            sparse_indices = indices.copy() # Make a deep copy of indices so that we may pop elements from sparse_indices later
            
            # Initialize the list to store groups
            sparse_batches = []
            
            # Calculate the centroid for each compact group as initial start for sparse groups
            # The idea is the centroids of each compact group are naturally sparse
            centroids = np.array([np.mean(pos[cbatch], axis=0) for cbatch in compact_batches])
            pairwise_distances = cdist(pos, pos) # Calculate the dist for ALL pos can keep the absolute index and skip the conversion between indexing
            
            used_indices = [] # This list stores the indices used for initialization of the sparse groups
            # Find the indices closest to the centroids of compact groups, these indices are the initial point for each sparse group
            for batch_idx in range(num_batch):
                distances = np.linalg.norm(pos_s - centroids[batch_idx], axis=1) # Note that this distances is only for selected pos (pos_s = pos[indices])
                closest_idx_s = np.argmin(distances) # closest_idx_s is the position of min distances
                closest_idx = indices[closest_idx_s] # closest_idx is the actual index that is closest to the centroid
                sparse_batches.append([closest_idx])
                used_indices.append(closest_idx_s)
            sparse_indices = np.delete(sparse_indices, used_indices) # Delete the used_indices after the entire loop, this helps keep indexing correct and consistent
            # Deleting elements in a loop would make indexing very challenging
            
            # Iterate through remaining points
            for idx in sparse_indices:
                min_distances = []
                # Iterate through groups
                for batch_idx in range(num_batch):
                    distances = pairwise_distances[sparse_batches[batch_idx], idx]
                    min_distances.append(np.min(distances))
                
                max_group_index = np.argmax(min_distances)

                # Add the point to the group with the farthest minimal distance
                sparse_batches[max_group_index].append(idx)
            
            # Final check because this procedure is fairly complicated
            flatten_indices = np.concatenate(sparse_batches)
            flatten_indices.sort()
            indices.sort()
            assert all(flatten_indices == indices), "Sorry, something went wrong with the sparse grouping, please try 'random' for now"
            print(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec")
            
            return sparse_batches

def save_results(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses):
    save_dict = make_save_dict(output_path, model, exp_params, source_params, loss_params, constraint_params, recon_params, loss_iters, iter_t, niter, batch_losses)

    torch.save(save_dict, os.path.join(output_path, f"model_iter{str(niter).zfill(4)}.pt"))

    imwrite(os.path.join(output_path, f"probe_amp_iter{str(niter).zfill(4)}.tif"), model.opt_probe.reshape(-1, model.opt_probe.size(-1)).t().abs().detach().cpu().numpy().astype('float32'))
    
    omode_occu = model.omode_occu
    omode      = model.opt_objp.size(0)
    zslice     = model.opt_objp.size(1)
    
    # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
    
    if omode == 1 and zslice == 1:
        imwrite(os.path.join(output_path, f"objp_iter{str(niter).zfill(4)}.tif"), model.opt_objp[0,0].detach().cpu().numpy().astype('float32'))
    elif omode == 1 and zslice > 1:
        imwrite(os.path.join(output_path, f"objp_zstack_iter{str(niter).zfill(4)}.tif"),       model.opt_objp[0,:].detach().cpu().numpy().astype('float32'))
        imwrite(os.path.join(output_path, f"objp_zsum_iter{str(niter).zfill(4)}.tif"),         model.opt_objp[0,:].sum(0).detach().cpu().numpy().astype('float32'))
    elif omode > 1 and zslice == 1:
        imwrite(os.path.join(output_path, f"objp_ostack_iter{str(niter).zfill(4)}.tif"),       model.opt_objp[:,0].detach().cpu().numpy().astype('float32'))
        imwrite(os.path.join(output_path, f"objp_omean_iter{str(niter).zfill(4)}.tif"),        model.opt_objp[:,0].mean(0).detach().cpu().numpy().astype('float32'))
        imwrite(os.path.join(output_path, f"objp_ostd_iter{str(niter).zfill(4)}.tif"),         model.opt_objp[:,0].std(0).detach().cpu().numpy().astype('float32'))
    else:
        imwrite(os.path.join(output_path, f"objp_4D_iter{str(niter).zfill(4)}.tif"),           model.opt_objp[:,:].detach().cpu().numpy().astype('float32'))
        imwrite(os.path.join(output_path, f"objp_ostack_zsum_iter{str(niter).zfill(4)}.tif"),  model.opt_objp[:,:].sum(1).detach().cpu().numpy().astype('float32'))
        imwrite(os.path.join(output_path, f"objp_omean_zstack_iter{str(niter).zfill(4)}.tif"), model.opt_objp[:,:].mean(0).detach().cpu().numpy().astype('float32'))

def imshift_single(img, shift, grid):
    """
    Generates a single shifted image from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shift (torch.Tensor): The shift to be applied to the image. It should be a (2,) tensor and as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from 0 to 1

    Outputs:
        shifted_img (torch.Tensor): The shifted image (..., Ny, Nx),

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so *[None]*ndim means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to (Nb, *img.shape), so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shift = shift[..., *[None]*(ndim-1)]                                              # Expand shifts to (2,1,1,...) so shifts.ndim = ndim+1
    grid = grid[:,*[None]*(ndim-2), ...]                                              # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+1
    shift_y, shift_x = shift[0], shift[1]                                             # shift_y, shift_x are (1,1,...) with ndim singletons, so the shift_y.ndim = ndim
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx + shift_y * ky))                   # w = (1,1,...,Ny,Nx) so w.ndim = ndim
    shifted_img = ifft2(ifftshift(fftshift(fft2(img), dim=(-2,-1)) * w, dim=(-2,-1))) # For real-valued input, take shifted_img.abs()
    return shifted_img

def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from 0 to 1

    Outputs:
        shifted_img (torch.Tensor): The batch of shifted images. It has an extra dimension than the input image, i.e., shape=(Nb, ..., Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so *[None]*ndim means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to (Nb, *img.shape), so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shifts = shifts[..., *[None]*ndim]                                                # Expand shifts to (Nb,2,1,1,...) so shifts.ndim = ndim+2
    grid = grid[:,*[None]*(ndim-1), ...]                                              # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+2
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]                                     # shift_y, shift_x are (Nb,1,1,...) with ndim singletons, so the shift_y.ndim = ndim+1
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim+1
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx + shift_y * ky))                   # w = (Nb, 1,1,...,Ny,Nx) so w.ndim = ndim+1
    shifted_img = ifft2(ifftshift(fftshift(fft2(img), dim=(-2,-1)) * w, dim=(-2,-1))) # For real-valued input, take shifted_img.abs()
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
        print(f'Real space probe extent = {dx*Npix:.4f} Ang')
    
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

def get_center_of_mass(image, corner_centered=False):
    """ Finds and returns the center of mass of an real-valued 2/3D tensor """
    # The expected input shape can be either (Ny, Nx) or (N, Ny, Nx)
    # The output center_y and center_x will be either (N,) or a scaler tensor
    # Note that for even-number sized arr (like [128,128]), even it's uniformly ones, the "center" would be between pixels like [63.5,63.5]
    # Note that the `corner_centered` flag idea is adapted from py4DSTEM, which is quite handy when we have corner-centered probe or CBED
    # https://github.com/py4dstem/py4DSTEM/blob/dev/py4DSTEM/process/utils/utils.py
    
    ndim = image.ndim
    assert ndim in [2, 3], f"image.ndim must be either 2 or 3, we've got {ndim}"
    
    # Create grid of coordinates
    device = image.device
    (ny, nx) = image.shape[-2:]

    if corner_centered:
        grid_y, grid_x = torch.meshgrid(torch.fft.fftfreq(ny, 1 / ny, device=device), torch.fft.fftfreq(nx, 1 / nx, device=device), indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    
    # Compute total intensity
    total_intensity = torch.sum(image, dim = (-2,-1)).mean()
    
    # Compute weighted sum of x and y coordinates
    center_y = torch.sum(grid_y * image, dim = (-2,-1)) / total_intensity
    center_x = torch.sum(grid_x * image, dim = (-2,-1)) / total_intensity
    
    return center_y, center_x

def get_blob_size(dx, blob, output='d90', plot_profile=False):
    import matplotlib.pyplot as plt
    """ Get the probe / blob size

    Args:
        dx (float): px size in Ang
        blob (array): the probe/blob image, note that we assume the input is already directly measurable and no squaring is needed, centered, and background free
        plot_profile (bool): Flag for plotting the profile or not 

    Returns:
        D50*dx: D50 in Ang
        D90*dx: D90 in Ang
        radius_rms*dx: RMS radius in Ang
        radial_profile: radially averaged profile
        radial_sum: radial profile without normalizing by the ring area
        fig: Line profile figure
    """
    def get_radial_profile(data, center):
        # The radial intensity is calculated up to the corners
        # So len(radialprofile) will be len(data)/sqrt(2)
        # The bin width is set to be the same with original data spacing (dr = dx)
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        radial_sum = tbin
        return radial_profile, radial_sum

    radial_profile, radial_sum = get_radial_profile(blob, (len(blob)//2, len(blob)//2))
    #print("sum(radial_sum) = %.5f " %(np.sum(radial_sum)))

    # Calculate the rms radius, in px
    x = np.arange(len(radial_profile))
    radius_rms = np.sqrt(np.sum(x**2*radial_profile*x)/np.sum(radial_profile*x))

    # Calculate FWHM
    
    HWHM = np.max(np.where((radial_profile / radial_profile.max()) >=0.5))
    
    # Calculate D50, D90
    cum_sum = np.cumsum(radial_sum)

    # R50, 90 without normalization
    R50 = np.min(np.where(cum_sum>=0.50*np.sum(radial_sum))[0])
    R90 = np.min(np.where(cum_sum>=0.90*np.sum(radial_sum))[0])
    R99 = np.min(np.where(cum_sum>=0.99*np.sum(radial_sum))[0])
    R995 = np.min(np.where(cum_sum>=0.995*np.sum(radial_sum))[0])
    R999 = np.min(np.where(cum_sum>=0.999*np.sum(radial_sum))[0])

    D50  = (2*R50+1)
    D90  = (2*R90+1)
    D99  = (2*R99+1)
    D995 = (2*R995+1)
    D999 = (2*R999+1)
    FWHM = (2*HWHM+1)

    if plot_profile:
        
        num_ticks = 11
        x = dx*np.arange(len(radial_profile))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Radially averaged profile")
        plt.margins(x=0, y=0)
        ax.plot(x, radial_profile/np.max(radial_profile), label='Radially averaged profile')
        #plt.plot(x, cum_sum, 'k--', label='Integrated current')
        plt.vlines(x=R50*dx, ymin=0, ymax=1, color="tab:orange", linestyle=":", label='R50') #Draw vertical lines at the data coordinate, in this case would be Ang.
        plt.vlines(x=R90*dx, ymin=0, ymax=1, color="tab:red", linestyle=":", label='R90')
        plt.vlines(x=HWHM*dx, ymin=0, ymax=1, color="tab:blue", linestyle=":", label='FWHM')
        plt.vlines(x=radius_rms*dx, ymin=0, ymax=1, color="tab:green", linestyle=":", label='Radius_RMS')
        plt.xticks(np.arange(num_ticks)*np.round(len(radial_profile)*dx/num_ticks, decimals = 1-int(np.floor(np.log10(len(radial_profile)*dx)))))
        ax.set_xlabel("Distance from blob center ($\AA$)")
        ax.set_ylabel("Normalized intensity")
        plt.legend()
        plt.show()

    if output == 'd50':
        out = D50*dx
    elif output =='d90':
        out =  D90*dx
    elif output =='d99':
        out =  D99*dx
    elif output =='d995':
        out =  D995*dx
    elif output =='d999':
        out =  D999*dx
    elif output =='radius_rms':
        out =  radius_rms*dx
    elif output =='FWHM':
        out =  FWHM*dx
    elif output =='radial_profile':
        out =  radial_profile
    elif output =='radial_sum':
        out =  radial_sum
    elif output =='fig':
        out =  fig
    else:
        raise KeyError(f"output ={output} not implemented!")
    
    if output not in ['radial_profile', 'radial_sum', 'fig']:
        print(f'{output} = {out/dx:.3f} px or {out:.3f} Ang')
    return out

def make_sigmoid_mask(Npix, relative_radius=2/3, relative_width=0.2):
    ''' Make a mask from circular sigmoid function '''    
    # relative_radius = 0.67 # This is the relative Nyquist frequency where the sigmoid = 0.5
    # relative_width  = 0.2 # This is the relative width (compared to full image) that y drops from 1 to 0
    
    def scaled_sigmoid(x, offset=0, scale=1):
        # If scale =  1, y drops from 1 to 0 between (-0.5,0.5), or effectively 1 px
        # If scale = 10, it takes roughly 10 px for y to drop from 1 to 0
        scaled_sigmoid = 1 / (1 + torch.exp((x-offset)/scale*10))
        return scaled_sigmoid
    
    ky = torch.linspace(-floor(Npix/2),ceil(Npix/2)-1,Npix)
    kx = torch.linspace(-floor(Npix/2),ceil(Npix/2)-1,Npix)
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')
    kR = torch.sqrt(grid_ky**2+grid_kx**2) # centered already
    sigmoid_mask = scaled_sigmoid(kR, offset=Npix/2*relative_radius, scale=relative_width*Npix)
    
    return sigmoid_mask

def get_rbf(cbeds, thresh=0.5):
    """ Utility function that returns an estimate of the radius of rbf from CBEDs """
    # cbeds: 3D array of (N,ky,kx) so that we can take an average 
    # thresh: 0.5 for FWHM, 0.1 for Full-width at 10th maximum
    cbed = cbeds.sum(0)
    line = cbed.max(0)
    indices = np.where(line > line.max()*thresh)[0]
    rbf = 0.5*(indices[-1]-indices[0])
    return rbf

###################################### ARCHIVE ##################################################

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

