## Define the loss function class with loss and regularizations
## Define the constraint class for iter-wist constraints
## Define the optimization loop related functions

from .utils import time_sync, make_sigmoid_mask, fftshift2, ifftshift2
import numpy as np
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import interpolate
import torch
from torch.fft import fft2, ifft2, fftn, ifftn, fftfreq

# The CombinedLoss takes a user-defined dict of loss_params, which specifies the state, weight, and param of each loss term
# The CBED related loss takes a parameter of dp_pow which raise the CBED with certain power, 
# usually 0.5 for loss_single and 0.2 for loss_pacbed to emphasize the diffuse background
# The obj-dependent regularization loss_sparse is using the objp_patches as input
# In this way it'll only calculate values within the ROI, so the edges of the object would not be included

class CombinedLoss(torch.nn.Module):
    """ Calculate the loss with regularization on the object phase patches for each batch """

    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = torch.nn.MSELoss(reduction='mean')

    def get_loss_single(self, model_CBEDs, measured_CBEDs):
        # Calculate loss_single
        # This loss function emulates the likelihood function of cbeds with Gaussian statistics (higher dose)
        # For exact Gaussian statistics, the dp_pow should be 0.5
        
        single_params = self.loss_params['loss_single']
        if single_params['state']:
            dp_pow = single_params.get('dp_pow', 0.5)
            data_mean = measured_CBEDs.pow(dp_pow).mean()
            loss_single = self.mse(model_CBEDs.pow(dp_pow), measured_CBEDs.pow(dp_pow))**0.5 / data_mean # Doing Normalized RMSE makes the value quite consistent between dp_pow 0.2-0.5.
            loss_single *= single_params['weight']
        else:
            loss_single = torch.tensor(0, dtype=torch.float32, device=self.device) # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        return loss_single
    
    def get_loss_poissn(self, model_CBEDs, measured_CBEDs):
        # Calculate loss_poissn
        # This loss function emulates the likelihood function of cbeds with Poisson statistics (low dose)
        # For exact Poisson statistics, the dp_pow should be 1
        # No need to worry about the CBED having most pixel value smaller than 1, CBED int scaling has no effect to the reconstruction
        # The eps in log is needed for numerical stability during optimization and to avoid negative infinite when the CBED intensity is approaching 0
        # Typical eps is within 1e-3 to 1e-9
        
        # function L = get_loglik(modF, aPsi)
        # modF2 = modF.^2; # exp
        # aPsi2 = aPsi.^2; # model
        # L = -(modF2 .* log(aPsi2+1e-6) - aPsi2) ;
        poissn_params = self.loss_params['loss_poissn']
        
        if poissn_params['state']:
            dp_pow = poissn_params.get('dp_pow', 1)
            eps = poissn_params.get('eps', 1e-6)
            data_mean = measured_CBEDs.pow(dp_pow).mean()
            loss_poissn = -torch.mean(measured_CBEDs.pow(dp_pow) * torch.log(model_CBEDs.pow(dp_pow) + eps) - model_CBEDs.pow(dp_pow)) / data_mean # Doing Normalized RMSE makes the value quite consistent between dp_pow 0.2-0.5.
            loss_poissn *= poissn_params['weight']
        else:
            loss_poissn = torch.tensor(0, dtype=torch.float32, device=self.device) # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        return loss_poissn
    
    def get_loss_pacbed(self, model_CBEDs, measured_CBEDs):
        # Calculate loss_pacbed
        pacbed_params = self.loss_params['loss_pacbed']
        if pacbed_params['state']:
            dp_pow = pacbed_params.get('dp_pow', 0.2)
            loss_pacbed = self.mse(model_CBEDs.mean(0).pow(dp_pow), measured_CBEDs.mean(0).pow(dp_pow))**0.5
            loss_pacbed *= pacbed_params['weight']
        else:
            loss_pacbed = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_pacbed
        
    def get_loss_sparse(self, objp_patches, omode_occu):
        # Calculate loss_sparse by considering the ln norm
        # For obj-dependent regularization terms, the omode contribution should be weighting the individual loss for each omode.
        # Scaling the obj value by its omode_occu would make non-linear loss like l2 dependent on # of omode.
        # Therefore, the proper way is to get a loss tensor L(obj) shaped (N, omode, Nz, Ny, Nx) and then do the voxel-wise mean across (N,:,Nz,Ny,Nx)
        # and lastly we do the weighted sum with omode_occu so that the loss value is not batch, object size, or omode dependent.
        sparse_params = self.loss_params['loss_sparse']
        if sparse_params['state']:
            ln_order = sparse_params['ln_order']
            loss_sparse = sparse_params['weight'] * (torch.mean(objp_patches.abs().pow(ln_order), dim=(0,2,3,4)).pow(1/ln_order) * omode_occu).sum()
        else:
            loss_sparse = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_sparse
    
    def get_loss_simlar(self, object_patches, omode_occu):
        # Calculate loss_simlar by calculating the similarity between different omodes
        # This loss term is specifically designed for regularizing omode by reducing the std of Gaussian_blurred / downsampled obj along the omode dimension
        # obja/p_patches = (N,omode,Nz,Ny,Nx) 
        simlar_params = self.loss_params['loss_simlar']
        if simlar_params['state']:
            obj_type     = simlar_params['obj_type']
            obj_blur_std = simlar_params['blur_std']
            scale_factor = simlar_params['scale_factor']
            obja_patches = object_patches[...,0]
            objp_patches = object_patches[...,1]
            temp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            
            if obj_type in ['amplitude', 'both']:
                if obj_blur_std is not None and obj_blur_std != 0:
                    # The torch stack and list comp is needed because gaussian_blur defaulted "replicate" padding, and for 2D padding it's only supporting 4D tenosr while the obja_patches is 5D
                    obja_patches = torch.stack([gaussian_blur(obja_patch, kernel_size=5, sigma=obj_blur_std) for obja_patch in obja_patches]) 
                if scale_factor is not None and any(scale != 1 for scale in scale_factor):
                    obja_patches = interpolate(obja_patches, scale_factor = scale_factor, mode = 'area')  
                temp_loss += (obja_patches * omode_occu[:,None,None,None]).std(1).mean()
            if obj_type in ['phase', 'both']:
                if obj_blur_std is not None and obj_blur_std != 0:
                    objp_patches = torch.stack([gaussian_blur(objp_patch, kernel_size=5, sigma=obj_blur_std) for objp_patch in objp_patches])
                if scale_factor is not None and any(scale != 1 for scale in scale_factor):
                    objp_patches = interpolate(objp_patches, scale_factor = scale_factor, mode = 'area')  
                temp_loss += (objp_patches * omode_occu[:,None,None,None]).std(1).mean()
            loss_simlar = simlar_params['weight'] * temp_loss
        else:
            loss_simlar = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_simlar
    
    def forward(self, model_CBEDs, measured_CBEDs, object_patches, omode_occu):
        losses = []
        losses.append(self.get_loss_single(model_CBEDs, measured_CBEDs))
        losses.append(self.get_loss_poissn(model_CBEDs, measured_CBEDs))
        losses.append(self.get_loss_pacbed(model_CBEDs, measured_CBEDs))
        losses.append(self.get_loss_sparse(object_patches[...,1], omode_occu))
        losses.append(self.get_loss_simlar(object_patches, omode_occu))
        total_loss = sum(losses)
        return total_loss, losses
    
class CombinedConstraint(torch.nn.Module):
    ''' Apply iteration-wise in-place constraints on the optimizable tensors '''
    
    def __init__(self, constraint_params, device='cuda:0'):
        super(CombinedConstraint, self).__init__()
        self.device = device
        self.constraint_params = constraint_params

    def apply_ortho_pmode(self, model, niter):
        ''' Apply orthogonality constraint to probe modes '''
        ortho_pmode_freq = self.constraint_params['ortho_pmode']['freq']
        if ortho_pmode_freq is not None and niter % ortho_pmode_freq == 0:
            model.opt_probe.data = orthogonalize_modes_vec(model.opt_probe, sort=True)
            probe_int = model.opt_probe.abs().pow(2)
            probe_pow = (probe_int.sum((1,2))/probe_int.sum()).detach().cpu().numpy().round(3)
            print(f"Apply ortho pmode constraint at iter {niter}, relative pmode power = {probe_pow}, probe int sum = {model.opt_probe.abs().pow(2).sum():.4f}")

    def apply_probe_mask_k(self, model, niter):
        ''' Apply probe amplitude constraint in Fourier space '''
        # Note that this will change the total probe intensity, please use this with `fix_probe_int`
        # Although the mask wouldn't change during the iteration, making a mask takes only ~0.5us on CPU so really no need to pre-calculate it
        # The sandwitch fftshift(fft(ifftshift(probe))) is needed to properly handle the complex probe without serrated phase
        # fft2 is for real->fourier, while fftshift2 is for corner->center
        
        probe_mask_k_freq = self.constraint_params['probe_mask_k']['freq']
        relative_radius  = self.constraint_params['probe_mask_k']['radius']
        relative_width   = self.constraint_params['probe_mask_k']['width']
        if probe_mask_k_freq is not None and niter % probe_mask_k_freq == 0:
            Npix = model.opt_probe.size(-1)
            mask = make_sigmoid_mask(Npix, relative_radius, relative_width).to(model.device)
            probe_k = fftshift2 (fft2(ifftshift2(model.opt_probe), norm='ortho')) # probe_k at center for later masking
            probe_r = fftshift2(ifft2(ifftshift2(mask * probe_k),  norm='ortho')) # probe_r at center. Note that the norm='ortho' is explicitly specified but not needed for a round-trip 
            model.opt_probe.data = probe_r
            print(f"Apply Fourier-space probe amplitude constraint at iter {niter}, probe int sum = {model.opt_probe.abs().pow(2).sum():.4f}")
    
    def apply_fix_probe_int(self, model, niter):
        ''' Apply probe intensity constraint '''
        # Note that the probe intensity fluctuation (std/mean) is typically only 0.5%, there's very little point to do a position-dependent probe intensity constraint
        # Therefore, a mean probe intensity is used here as the target intensity
        fix_probe_int_freq = self.constraint_params['fix_probe_int']['freq']
        if fix_probe_int_freq is not None and niter % fix_probe_int_freq == 0: 
            current_amp = model.opt_probe.abs().pow(2).sum().pow(0.5)
            target_amp  = model.probe_int_sum**0.5   
            model.opt_probe.data = model.opt_probe * target_amp/current_amp
            print(f"Apply fix probe int constraint at iter {niter}, probe int sum = {model.opt_probe.abs().pow(2).sum():.4f}")
            
    def apply_obj_blur(self, model, niter):
        ''' Apply Gaussian blur to object, this only applies to the last 2 dimension (...,H,W) '''
        # Note that it's not clear whether applying blurring after every iteration would ever reach a steady state
        # However, this is at least similar to PtychoShelves' eng. reg_mu
        obj_blur_freq = self.constraint_params['obj_blur']['freq']
        obj_type      = self.constraint_params['obj_blur']['obj_type']
        obj_blur_std  = self.constraint_params['obj_blur']['std']
        if obj_blur_freq is not None and niter % obj_blur_freq == 0 and obj_blur_std !=0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = gaussian_blur(model.opt_obja, kernel_size=5, sigma=obj_blur_std)
                print(f"Apply lateral (y,x) Gaussian blur with std = {obj_blur_std} px on obja at iter {niter}")
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = gaussian_blur(model.opt_objp, kernel_size=5, sigma=obj_blur_std)
                print(f"Apply lateral (y,x) Gaussian blur with std = {obj_blur_std} px on objp at iter {niter}")
                
    def apply_kr_filter(self, model, niter):
        ''' Apply kr Fourier filter constraint on object '''
        # Note that the `kr_filter` is applied on stacked 2D FFT of object, so it's applying on (omode,z,ky,kx)
        # The kr filter is similar to a top-hat, so it's more like a cut-off, instead of the weak lateral Gaussian blurring (alpha) included in the `kz_filter` 
        kr_filter_freq   = self.constraint_params['kr_filter']['freq']
        obj_type         = self.constraint_params['kr_filter']['obj_type']
        relative_radius  = self.constraint_params['kr_filter']['radius']
        relative_width   = self.constraint_params['kr_filter']['width']
        if kr_filter_freq is not None and niter % kr_filter_freq == 0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kr_filter(model.opt_obja, relative_radius, relative_width)
                print(f"Apply kr_filter constraint with kr_radius = {relative_radius} on obja at iter {niter}")
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kr_filter(model.opt_objp, relative_radius, relative_width)
                print(f"Apply kr_filter constraint with kr_radius = {relative_radius} on objp at iter {niter}")
        
    def apply_kz_filter(self, model, niter):
        ''' Apply kz Fourier filter constraint on object '''
        # Note that the `kz_filter`` behaves differently for 'amplitude' and 'phase', see `kz_filter` implementaion for details
        kz_filter_freq         = self.constraint_params['kz_filter']['freq']
        obj_type               = self.constraint_params['kz_filter']['obj_type']
        beta_regularize_layers = self.constraint_params['kz_filter']['beta']
        alpha_gaussian         = self.constraint_params['kz_filter']['alpha']
        if kz_filter_freq is not None and niter % kz_filter_freq == 0:
            if obj_type in ['amplitude', 'both']:
                model.opt_obja.data = kz_filter(model.opt_obja, beta_regularize_layers, alpha_gaussian, obj_type='amplitude')
                print(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on obja at iter {niter}")
            if obj_type in ['phase', 'both']:
                model.opt_objp.data = kz_filter(model.opt_objp, beta_regularize_layers, alpha_gaussian, obj_type='phase')
                print(f"Apply kz_filter constraint with beta = {beta_regularize_layers} on objp at iter {niter}")
    
    def apply_obja_thresh(self, model, niter):
        ''' Apply thresholding on obja at voxel level '''
        # Although there's a lot of code repitition with `apply_postiv`, phase positivity itself is important enough as an individual operation
        obja_thresh_freq = self.constraint_params['obja_thresh']['freq']
        relax            = self.constraint_params['obja_thresh']['relax']
        thresh           = self.constraint_params['obja_thresh']['thresh']
        if obja_thresh_freq is not None and niter % obja_thresh_freq == 0: 
            model.opt_obja.data = relax * model.opt_obja + (1-relax) * model.opt_obja.clamp(min=thresh[0], max=thresh[1])
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_clamp))' if relax != 0 else 'hard'
            print(f"Apply {relax_str} threshold constraint with thresh = {np.round(thresh,5)} on obja at iter {niter}")

    def apply_objp_postiv(self, model, niter):
        ''' Apply positivity constraint on objp at voxel level '''
        # Note that this `relax` is defined oppositly to PtychoShelves's `positivity_constraint_object` in `ptycho_solver`. 
        # Here, relax=1 means fully relaxed and essentially no constraint.
        objp_postiv_freq = self.constraint_params['objp_postiv']['freq']
        relax            = self.constraint_params['objp_postiv']['relax'] 
        if objp_postiv_freq is not None and niter % objp_postiv_freq == 0: 
            model.opt_objp.data = relax * model.opt_objp + (1-relax) * model.opt_objp.clamp(min=0)
            relax_str = f'relaxed ({relax}*obj + ({1-relax}*obj_postiv))' if relax != 0 else 'hard'
            print(f"Apply {relax_str} positivity constraint on objp at iter {niter}")           
        
    def apply_tilt_smooth(self, model, niter):
        ''' Apply Gaussian blur to object tilts '''
        # Note that the smoothing is applied along the last 2 axes, which are scan dimensions, so the unit of std is "scan positions"
        # Besides, the relative position of the obj_tilts are neglected for simplicity
        tilt_smooth_freq = self.constraint_params['tilt_smooth']['freq']
        tilt_smooth_std  = self.constraint_params['tilt_smooth']['std']
        N_scan_slow = model.N_scan_slow
        N_scan_fast = model.N_scan_fast
        if tilt_smooth_freq is not None and niter % tilt_smooth_freq == 0 and tilt_smooth_std !=0:
            obj_tilts = (model.opt_obj_tilts.reshape(N_scan_slow, N_scan_fast, 2)).permute(2,0,1)
            model.opt_obj_tilts.data = gaussian_blur(obj_tilts, kernel_size=5, sigma=tilt_smooth_std).permute(1,2,0).reshape(-1,2)
            print(f"Apply Gaussian blur with std = {tilt_smooth_std} scan positions on obj_tilts at iter {niter}")
    
    def forward(self, model, niter):
        # Apply in-place constraints if niter satisfies the predetermined frequency
        # Note that the if check blocks are included in each apply methods so that it's cleaner, and I can print the info with niter
        
        with torch.no_grad():
            # Probe constraints
            self.apply_ortho_pmode  (model, niter)
            self.apply_probe_mask_k (model, niter)
            self.apply_fix_probe_int(model, niter)
            # Object constraints
            self.apply_obj_blur     (model, niter)
            self.apply_kr_filter    (model, niter)
            self.apply_kz_filter    (model, niter)
            self.apply_obja_thresh  (model, niter)
            self.apply_objp_postiv  (model, niter)
            # Local tilt constraint
            self.apply_tilt_smooth  (model, niter)

def ptycho_recon(batches, model, optimizer, loss_fn, constraint_fn, niter):
    ''' Perform 1 iteration of the ptycho reconstruciton in the optimization loop '''
    batch_losses = {name: [] for name in loss_fn.loss_params.keys()}
    start_iter_t = time_sync()
    
    # Start mini-batch optimization
    for batch_idx, batch in enumerate(batches):
        start_batch_t = time_sync()
        optimizer.zero_grad()
        model_CBEDs, object_patches = model(batch)
        measured_CBEDs = model.get_measurements(batch)
        loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, object_patches, model.omode_occu)
        loss_batch.backward()
        optimizer.step() # batch update
        batch_t = time_sync() - start_batch_t

        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            batch_losses[loss_name].append(loss_value.detach().cpu().numpy())

        if batch_idx in np.linspace(0, len(batches)-1, num=6, dtype=int):
            print(f"Done batch {batch_idx+1} in {batch_t:.3f} sec")
    
    # Apply iter-wise constraint
    constraint_fn(model, niter)
    
    iter_t = time_sync() - start_iter_t
    return batch_losses, iter_t

def loss_logger(batch_losses, niter, iter_t):
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ', '.join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    print(f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, "
          f"in {iter_t // 60} min {iter_t % 60:03f} sec")
    loss_iter = sum(avg_losses.values())
    return loss_iter    

def kr_filter(obj, radius, width):
    ''' Apply kr_filter using the 2D sigmoid filter '''
    
    # Create the filter function W, note that the W has to be corner-centered
    Ny, Nx = obj.shape[-2:]
    mask = make_sigmoid_mask(min(Ny,Nx), radius, width).to(obj.device)
    W = ifftshift2(interpolate(mask[None,None,], size=(Ny,Nx))).squeeze() # interpolate needs 2 additional dimension (N,C,...) for the input than the output dimension
        
    # Filter the obj with filter funciton Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifft2(fft2(obj) * W[None,None,])) # Apply fft2/ifft2 for only the r(y,x) dimension so the omode and z would be broadcasted
    
    return fobj

def kz_filter(obj, beta_regularize_layers=1, alpha_gaussian=1, obj_type='phase'):
    ''' Apply kz_filter using the arctan filter '''
    # Note: Calculate force of regularization based on the idea that DoF = resolution^2/lambda
        
    device = obj.device
    
    # Generate 1D grids along each dimension
    Npix = obj.shape[1:]
    kz = fftfreq(Npix[0]).to(device) 
    ky = fftfreq(Npix[1]).to(device) 
    kx = fftfreq(Npix[2]).to(device) 
    
    # Generate 3D coordinate grid using meshgrid
    grid_kz, grid_ky, grid_kx = torch.meshgrid(kz, ky, kx, indexing='ij')

    # Create the filter function Wa. W and Wa is exactly the same as PtychoShelves for now
    W = 1 - torch.atan((beta_regularize_layers * torch.abs(grid_kz) / torch.sqrt(grid_kx**2 + grid_ky**2 + 1e-3))**2) / (torch.pi/2)
    Wa = W * torch.exp(-alpha_gaussian * (grid_kx**2 + grid_ky**2))

    # Filter the obj with filter funciton Wa, take the real part because Fourier-filtered obj could contain negative values    
    fobj = torch.real(ifftn(fftn(obj, dim=(1,2,3)) * Wa[None,], dim=(1,2,3))) # Apply fftn/ifftn for only spatial dimension so the omode would be broadcasted
    
    if obj_type == 'amplitude':
        fobj = 1+0.9*(fobj-1) # This is essentially a soft obja threshold constraint built into the kz_filter routine for obja
        
    return fobj

def orthogonalize_modes_vec(modes, sort = False):
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

    _, evecs = torch.linalg.eig(A)
   
    # Matrix-multiplication version (N,N) @ (N,YX) = (N,YX)
    ortho_modes = torch.matmul(evecs.t(), modes_reshaped).reshape(input_shape)

    # sort modes by their contribution
    if sort:
        modes_int =  ortho_modes.abs().pow(2).sum(tuple(range(1,ortho_modes.ndim))) # Sum every but 1st dimension
        _, indices = torch.sort(modes_int, descending=True)
        ortho_modes = ortho_modes[indices]
        
    return ortho_modes