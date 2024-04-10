## Define the loss function class with loss and regularizations
## Define the constraint class for iter-wist constraints
## Define the optimization loop related functions

from .utils import time_sync
from math import ceil, floor
import numpy as np
from torchmetrics.image import TotalVariation
import torch

# This is a current working version (2024.04.02) of the CombinedLoss class
# I cleaned up the archived versions and slightly renamed the objects and variables for clarity

# The CombinedLoss takes a user-defined dict of loss_params, which specifies the state, weight, and param of each loss term
# Currently there're loss_single, loss_pacbed, loss_tv, loss_l1, loss_l2, and loss_postiv available
# The CBED related loss takes a parameter of dp_pow which raise the CBED with certain power, 
# usually 0.5 for loss_single and 0.2 for loss_pacbed to emphasize the diffuse background
# The obj-dependent regularizations (TV, L1, L2, postiv) are using the objp_patches as input
# In this way it'll only calculate values within the ROI, so the edges of the object would not be included

class CombinedLoss(torch.nn.Module):
    """ Calculate the loss with regularization on the object phase patches for each batch """
    
    # Example usage:
    # with torch.no_grad():
    #     loss_fn = CombinedLoss(loss_params, device=DEVICE)
    #     np.random.seed(42)
    #     indices = np.random.randint(0,N_max,48)
    #     model_CBEDs, objp_patches = model(indices)
    #     measured_CBEDs = measurements[indices]
    #     loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)
    # print(losses)
    
    # model/measured CBEDs (N,Ky,Kx), float32 tensor
    # objp_patches (N, omode, Nz, Ny, Nx), float32 tensor
    # omode_occu (omode), float32 tensor
    
    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.tv = TotalVariation(reduction='mean').to(device)

    def forward(self, model_CBEDs, measured_CBEDs, objp_patches, omode_occu):
        losses = []

        # Calculate loss_single
        single_params = self.loss_params['loss_single']
        if single_params['state']:
            dp_pow = single_params.get('dp_pow', 0.5)
            data_mean = measured_CBEDs.pow(dp_pow).mean()
            loss_single = self.mse(model_CBEDs.pow(dp_pow), measured_CBEDs.pow(dp_pow))**0.5 / data_mean # Doing Normalized RMSE makes it quite stable between dp_pow 0.2-0.5.
            loss_single *= single_params['weight']
        else:
            loss_single = torch.tensor(0, dtype=torch.float32, device=self.device) # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        losses.append(loss_single)

        # Calculate loss_pacbed
        pacbed_params = self.loss_params['loss_pacbed']
        if pacbed_params['state']:
            dp_pow = pacbed_params.get('dp_pow', 0.2)
            loss_pacbed = self.mse(model_CBEDs.mean(0).pow(dp_pow), measured_CBEDs.mean(0).pow(dp_pow))**0.5
            loss_pacbed *= pacbed_params['weight']
        else:
            loss_pacbed = torch.tensor(0, dtype=torch.float32, device=self.device)
        losses.append(loss_pacbed)
        
        # For obj-dependent regularization terms, the omode contribution should be weighting the individual loss for each omode.
        # Scaling the obj value by its omode_occu would make non-linear loss like l2 and tv value dependent on # of omode.
        # Therefore, the proper way is to get a loss tensor L(obj) shaped (N, omode, Nz, Ny, Nx) and then do the voxel-wise mean across (N,:,Nz,Ny,Nx)
        # and lastly we do the weighted sum with omode_occu so that the loss value is not batch, object size, or omode dependent.
        
        # Calculate loss_tv
        # https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html 
        # This TV only applies to the last 2 dim (N,C,H,W), so there's no z-direction measurement in our case with objp_patches (N, omode, Nz, Ny, Nx)
        tv_params = self.loss_params['loss_tv']
        if tv_params['state']:
            # Do the weighted sum along omode and normalize by number of z slices because the TV is summing over C and averaging over N.
            loss_tv = tv_params['weight'] * sum([self.tv(objp_patches[:,i])*omode_occu[i] for i in range(len(omode_occu))]) / objp_patches.shape[2] 
        else:
            loss_tv = torch.tensor(0, dtype=torch.float32, device=self.device)
        losses.append(loss_tv)

        # Calculate loss_l1
        l1_params = self.loss_params['loss_l1']
        if l1_params['state']:
            loss_l1 = l1_params['weight'] * (torch.mean(objp_patches.abs(), dim=(0,2,3,4)) * omode_occu).sum()
        else:
            loss_l1 = torch.tensor(0, dtype=torch.float32, device=self.device)
        losses.append(loss_l1)

        # Calculate loss_l2
        l2_params = self.loss_params['loss_l2']
        if l2_params['state']:
            loss_l2 = l2_params['weight'] * (torch.mean(objp_patches.pow(2), dim=(0,2,3,4)) * omode_occu).sum()
        else:
            loss_l2 = torch.tensor(0, dtype=torch.float32, device=self.device)
        losses.append(loss_l2)

        # Calculate loss_postiv
        postiv_params = self.loss_params['loss_postiv']
        if postiv_params['state']:
            loss_postiv = postiv_params['weight'] * (torch.mean(torch.relu(-objp_patches), dim=(0,2,3,4)) * omode_occu).sum()
        else:
            loss_postiv = torch.tensor(0, dtype=torch.float32, device=self.device)
        losses.append(loss_postiv)

        
        total_loss = sum(losses)
        return total_loss, losses
    
class CombinedConstraint(torch.nn.Module):
    ''' Apply iteration-wise in-place constraints on the optimizable tensors '''
    def __init__(self, constraint_params, device='cuda:0'):
        super(CombinedConstraint, self).__init__()
        self.device = device
        self.constraint_params = constraint_params

    def forward(self, model, iter):
        
        # Apply in-place constraints 
        with torch.no_grad():

            # Apply positivity constraint
            postiv_freq = self.constraint_params['postiv']['freq']
            if postiv_freq is not None and iter % postiv_freq == 0: # Check if iter % freq == 0. Since iter starts at 1 
                print(f"Apply postiv constraint at iter {iter}")
                model.opt_objp.clamp_(min=0)
            
            # Apply orthogonality constraint to probe modes
            ortho_pmode_freq = self.constraint_params['ortho_pmode']['freq']
            if ortho_pmode_freq is not None and iter % ortho_pmode_freq == 0:
                print(f"Apply ortho pmode constraint at iter {iter}")
                model.opt_probe.data = orthogonalize_modes_vec(model.opt_probe)

            # Apply orthogonality constraint to pmode
            ortho_omode_freq = self.constraint_params['ortho_omode']['freq']
            if ortho_omode_freq is not None and iter % ortho_omode_freq == 0:
                print(f"Apply ortho omode constraint at iter {iter}")
                model.opt_objp.data = orthogonalize_modes_vec(model.opt_objp)
                
            # Apply kz filter constraint
            kz_filter_freq = self.constraint_params['kz_filter']['freq']
            beta_regularize_layers = self.constraint_params['kz_filter']['beta']
            alpha_gaussian = self.constraint_params['kz_filter']['alpha']
            z_pad = self.constraint_params['kz_filter']['z_pad']
            if kz_filter_freq is not None and iter % kz_filter_freq == 0:
                print(f"Apply kz_filter constraint at iter {iter}")
                model.opt_objp.data = kz_filter(model.opt_objp, beta_regularize_layers, alpha_gaussian, z_pad)
        return

def batch_update(batch, model, optimizer, loss_fn):
    ''' Optimization closure that performs 1 mini-batch update '''
    start_batch_t = time_sync()
    optimizer.zero_grad()
    model_CBEDs, objp_patches = model(batch)
    measured_CBEDs = model.get_measurements(batch)
    loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)
    loss_batch.backward()
    optimizer.step() # batch update
    batch_t = time_sync() - start_batch_t
    return losses, batch_t

def ptycho_recon(batches, model, optimizer, loss_fn, constraint_fn, iter):
    ''' Perform 1 iteration of the ptycho reconstruciton in the optimization loop '''
    batch_losses = {name: [] for name in loss_fn.loss_params.keys()}
    start_iter_t = time_sync()
    
    # Start mini-batch optimization
    for batch_idx, batch in enumerate(batches):
        losses, batch_t = batch_update(batch, model, optimizer, loss_fn)

        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            batch_losses[loss_name].append(loss_value.detach().cpu().numpy())

        if batch_idx in np.linspace(0, len(batches)-1, num=6, dtype=int):
            print(f"Done batch {batch_idx+1} in {batch_t:.3f} sec")
    
    # Apply iter-wise constraint
    constraint_fn(model, iter)
    
    iter_t = time_sync() - start_iter_t
    return batch_losses, iter_t

def loss_logger(batch_losses, iter, iter_t):
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    print(f"Iter: {iter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, "
          f"in {iter_t // 60} min {iter_t % 60:03f} sec")
    loss_iter = sum(avg_losses.values())
    return loss_iter    

def kz_filter(obj, beta_regularize_layers=1, alpha_gaussian=1, z_pad=None):
    # Calculate force of regularization based on the idea that DoF = resolution^2/lambda
    
    # Pad the tensor with zeros
    if z_pad is not None:
        obj = torch.nn.functional.pad(obj.permute(0,2,3,1), (z_pad,z_pad)).permute(0,3,1,2) # The padding only applies to dimensions from last to front so permute is needed
    
    device = obj.device
    
    # Generate 1D grids along each dimension
    Npix = obj.shape[1:]
    kz = torch.fft.fftfreq(Npix[0]).to(device) 
    ky = torch.fft.fftfreq(Npix[1]).to(device) 
    kx = torch.fft.fftfreq(Npix[2]).to(device) 
    
    # Generate 3D coordinate grid using meshgrid
    grid_kz, grid_ky, grid_kx = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Create the filter function Wa
    W = 1 - torch.atan((beta_regularize_layers * torch.abs(grid_kz) / torch.sqrt(grid_kx**2 + grid_ky**2 + 1e-20))**2) / (torch.pi/2)
    Wa = W * torch.exp(-alpha_gaussian * (grid_kx**2 + grid_ky**2))

    # Filter the obj with filter funciton Wa    
    fobj = torch.real(torch.fft.ifftn(torch.fft.fftn(obj, dim=(1,2,3)) * Wa[None,], dim=(1,2,3))) # Apply fftn/ifftn for only spatial dimension so the omode would be broadcasted
    
    # Select the original z-range before padding
    if z_pad is not None:
        fobj = fobj[:,z_pad:-z_pad,:,:]
    
    return fobj

def orthogonalize_modes_vec(modes):
    ''' orthogonalize the modes using SVD'''
    # Input:
    #   modes: input function with multiple modes
    # Output:
    #   ortho_modes: 
    # Note:
    #   This function is a highly vectorized PyTorch implementation of `ptycho\+core\probe_modes_ortho.m` from PtychoShelves
    #   It's numerically equivalent with the following for-loop version but is ~ 10x faster on small complex64 tensors (10,164,164) 
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The expected shape of `modes` input is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   If you check the orthoganality of each mode, make sure to change the input into complex128 or to modify the default tolerance of torch.allclose.
    #   Lastly, this operation could probably be so much faster with some proper vectorization
    
    # # Execute iter-wise constraints
    # if model.opt_probe.size(0) >1 and model.optimizable_tensors['probe'].requires_grad:
    #     with torch.no_grad():
    #         print("Orthogonalizing probe modes")
    #         model.opt_probe.data = orthogonalize_modes(model.opt_probe)

    #input_shape = modes.shape # input_shape could be either (N,Y,X) or (N,Z,Y,X)
    
    if modes.dtype != torch.complex64:
        modes = torch.complex(modes, torch.zeros_like(modes))
    input_shape = modes.shape
    modes_reshaped = modes.reshape(input_shape[0], -1) # Reshape modes to have a shape of (Nmode, X*Y)
    A = torch.matmul(modes_reshaped, modes_reshaped.t()) # A = M M^T
    
    evals, evecs = torch.linalg.eig(A)

    # sort modes by their contribution
    evals_abs = torch.abs(evals)
    _, indices = torch.sort(evals_abs, descending=True)
    evecs = evecs[:, indices]

    # # Vectorized sum version 
    # ortho_modes_reshaped = modes_reshaped[:,None,:] * evecs[:, :, None] # modes_reshaped[:,None,:] = (N,1,YX), evecs[:, :, None] = (N,N,1)
    # ortho_modes = torch.sum(ortho_modes_reshaped.view(N,N,Y,X), dim=0) # Reshape ortho_modes_reshaped from (N,N,YX) to (N, N, Y, X), and sum along the first dimension to get the final orthogonalized modes
    
    # Matrix-multiplication version (N,N) @ (N,YX) = (N,YX)
    ortho_modes = torch.matmul(evecs.t(), modes_reshaped).reshape(input_shape)
    
    return ortho_modes

def orthogonalize_modes_loop(modes):
    ''' Similar implementation of SVD decomposition with PtychoShelves'''
    # (N,Y,X) = modes.shape
    # modes_reshaped = modes.view(N, -1) # Reshape modes to have a shape of (Nmode, X*Y)
    # A = torch.matmul(modes_reshaped, modes_reshaped.t()) # A = M M^T
    
        # calculate M M* and its eigenvectors
    N = modes.size(0)
    A = torch.zeros(N, N, dtype=modes.dtype, device=modes.device)
    for ii in range(N):
        p2 = modes[ii]
        for jj in range(N):
            p1 = modes[jj]
            A[ii, jj] = torch.sum(torch.dot(p2.view(-1), p1.view(-1)))
    
    evals, evecs = torch.linalg.eig(A)

    # sort modes by their contribution
    evals_abs = torch.abs(evals)
    _, indices = torch.sort(evals_abs, descending=True)
    evecs = evecs[:, indices]

    # orthogonalize modes
    ortho_modes = torch.zeros_like(modes)
    for jj in range(N):
        for ii in range(N):
            ortho_modes[jj] += modes[ii] * evecs[ii, jj]
    return ortho_modes