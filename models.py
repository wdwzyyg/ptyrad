## Defining PtychoAD class for the optimization object

from forward import  multislice_forward_model_batch_all
from utils import cplx_from_np, prepare_stack_transform, imshift_batch
import torch
from torchvision.transforms import v2

# This is a current working version (2024.03.20) implementation of the PtychoAD class
# It applies the sub-px shift using a Fourier shift theorem approach for the probes.
# Complex probe is naturally supported and the phase is smoothly rolled without issue.
# This implementation also pre-calculates the grid, and simultaneously process all the probe shifts in a single batch.
# A vectorized object cropping get_obj_ROI_vec is also provided that is significantly faster.
class PtychoAD(torch.nn.Module):
    def __init__(self, init_obj, init_probe, init_crop_pos, init_probe_pos_shifts, H, lr_params=None, device='cuda:0'):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obj = cplx_from_np(init_obj, cplx_type="amp_phase", ndim=-1).to(device)
            self.opt_probe = torch.tensor(init_probe, dtype=torch.complex64, device=device)  
            self.opt_probe_pos_shifts = torch.tensor(init_probe_pos_shifts, device=device)
            self.H = torch.tensor(H, dtype=torch.complex64, device=device)
            self.shift_probes = (lr_params['probe_pos_shifts'] != 0) # Set shift_probes to False if lr_params['probe_pos_shifts'] = 0
            
            Ny, Nx = init_probe.shape[-2:]
            ry, rx = torch.meshgrid(torch.arange(Ny, device=device), torch.arange(Nx, device=device), indexing='ij')
            self.shift_probes_grid = torch.stack([ry/Ny, rx/Nx], dim=0)
            # Create the grid for obj_ROI in a vectorized approach
            # ry is the y-grid (Ny,Nx), by adding the y coordinates from init_crop_pos (N,1) in a broadcast way, it becomes (N,Ny,Nx)
            # Stacking the modified ry and rx at the last dimension, we get obj_ROI_grid = (N,Ny,Nx,2)
            self.obj_ROI_grid = torch.stack([ry[None,:,:] + torch.tensor(init_crop_pos[:, None, None, 0], device=device), 
                                             rx[None,:,:] + torch.tensor(init_crop_pos[:, None, None, 1], device=device)], dim=-1)
                        
            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obj': self.opt_obj,
                'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }

            self.optimizer_params = []
            if lr_params:
                for param_name, lr in lr_params.items():
                    if param_name in self.optimizable_tensors:
                        self.optimizable_tensors[param_name].requires_grad = (lr != 0)  # Set requires_grad based on learning rate
                        if lr != 0:
                            self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                    else:
                        print(f"Warning: '{param_name}' is not a valid parameter name.")

            print('PtychoAD major variables:')
            for name, tensor in self.optimizable_tensors.items():
                print(f"{name}: {tensor.shape}, {tensor.dtype}, device:{tensor.device}, grad:{tensor.requires_grad}, lr:{lr_params[name]:.0e}")
            
    def get_obj_ROI(self, indices):
        """ Get object ROI with integer coordinates """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        object_patches = self.opt_obj[:,:,self.obj_ROI_grid[indices,...,0],self.obj_ROI_grid[indices,...,1],:].permute(2,0,1,3,4,5)
        return object_patches
    
    def get_probes(self, indices):
        """ Get probes for each position """
        # If you're not trying to optimize probe positions, there's not much point using sub-px shifted stationary probes
        # This function will return a single probe when self.shift_probes = False,
        # and would only be returning multiple sub-px shifted probes if you're optimizing self.opt_probe_pos_shifts

        if self.shift_probes:
            probes = imshift_batch(self.opt_probe, shifts = self.opt_probe_pos_shifts[indices], grid = self.shift_probes_grid)
        else:
            probes = self.opt_probe[None,...] # Extend a singleton N dimension, essentially using same probe for all samples
        
        return probes
        
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed as an array and representing the whole batch
        
        object_patches = self.get_obj_ROI(indices)
        probes = self.get_probes(indices)
        dp_fwd = multislice_forward_model_batch_all(object_patches, probes, self.H)
        
        return dp_fwd

# This is a temporary archive version (2024.03.20) implementation of the PtychoAD class
# It applies the sub-px shift using a Fourier shift theorem approach for the probes.
# Complex probe is naturally supported and the phase is smoothly rolled without issue.
# This implementation also pre-calculates the grid, and simultaneously process all the probe shifts in a single batch.
# A vectorized object cropping get_obj_ROI_vec is also provided that is significantly faster.
class PtychoAD_vec(torch.nn.Module):
    def __init__(self, init_obj, init_probe, init_crop_pos, init_probe_pos_shifts, H, lr_params=None, device='cuda:0'):
        super(PtychoAD_vec, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obj = cplx_from_np(init_obj, cplx_type="amp_phase", ndim=-1).to(self.device)
            self.opt_probe = torch.tensor(init_probe, dtype=torch.complex64).to(self.device)  
            self.opt_probe_pos_shifts = torch.tensor(init_probe_pos_shifts, device=self.device)
            self.crop_pos = torch.tensor(init_crop_pos, dtype=torch.int32, device=self.device)
            self.H = torch.tensor(H, dtype=torch.complex64, device=self.device)
            self.roi_shape = init_probe.shape[-2:]
            self.shift_probes = (lr_params['probe_pos_shifts'] != 0) # Set shift_probes to False if lr_params['probe_pos_shifts'] = 0
            Ny, Nx = self.roi_shape[0], self.roi_shape[1]
            ry, rx = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing='ij')
            self.shift_probes_grid = torch.stack([ry/Ny, rx/Nx], dim=0).to(self.device)
            self.crop_grid = torch.stack([ry, rx]).to(self.device)
            
                
            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obj': self.opt_obj,
                'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }

            self.optimizer_params = []
            if lr_params:
                for param_name, lr in lr_params.items():
                    if param_name in self.optimizable_tensors:
                        self.optimizable_tensors[param_name].requires_grad = (lr != 0)  # Set requires_grad based on learning rate
                        if lr != 0:
                            self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                    else:
                        print(f"Warning: '{param_name}' is not a valid parameter name.")

            print('PtychoAD major variables:')
            for name, tensor in self.optimizable_tensors.items():
                print(f"{name}: {tensor.shape}, {tensor.dtype}, device:{tensor.device}, grad:{tensor.requires_grad}, lr:{lr_params[name]:.0e}")
            
    # def get_obj_ROI(self, indices):
    #     """ Get object ROI with integer coordinates """
    #     # It's strongly recommended to do integer version of get_obj_ROI
    #     # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
    #     # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
    #     height, width  = self.roi_shape[0], self.roi_shape[1]
    #     object_patches = torch.zeros((len(indices), *self.opt_obj.shape[:2], height, width, 2)).to(self.device)

    #     for i, idx in enumerate(indices):
    #         height_start, height_end = self.crop_pos[idx,0], self.crop_pos[idx,0] + height
    #         width_start,  width_end  = self.crop_pos[idx,1], self.crop_pos[idx,1] + width
    #         object_patch      =  self.opt_obj[:, :, height_start:height_end, width_start:width_end, :] # object_patch (omode, D,H,W,2), 2 for the amp/phase channel
    #         object_patches[i] = object_patch

    #     return object_patches

    def get_obj_ROI_vec(self, indices):
        """ Get object ROI with integer coordinates using vectorized operation """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        height_indices = self.crop_grid[0,None,:,:] + self.crop_pos[indices, None, None, 0]
        width_indices  = self.crop_grid[1,None,:,:] + self.crop_pos[indices, None, None, 1]
        object_patches = self.opt_obj[:,:,height_indices,width_indices,:].permute(2,0,1,3,4,5)

        return object_patches
    
    def get_probes(self, indices):
        """ Get probes for each position """
        # This function will return a single probe when self.shift_probes = False,
        # and would only be returning multiple sub-px shifted probes if you're optimizing self.opt_probe_pos_shifts

        if self.shift_probes:
            probes = imshift_batch(self.opt_probe, shifts = self.opt_probe_pos_shifts[indices], grid = self.shift_probes_grid)
        else:
            probes = self.opt_probe[None,...] # Extend a singleton N dimension, essentially using same probe for all samples
        
        return probes
        
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed as an array and representing the whole batch
        
        object_patches = self.get_obj_ROI_vec(indices)
        probes = self.get_probes(indices)
        dp_fwd = multislice_forward_model_batch_all(object_patches, probes, self.H)
        
        return dp_fwd

# This is an archived version (2024.03.19) implementation of the PtychoAD class
# It calculates all diffraction patterns in a batch and uses the affine transformation built-in from torchvision
# for sub-px probe shift that is significantly faster than the hand-crafted STN approach below.
# Unfortunately, the sub-px shift isn't ideal and introduce significant artifact in the CBED because
# the phase of the probe is not handled correctly. Shifting complex functions take a lot more consideration,
class PtychoAD_v2(torch.nn.Module):
    def __init__(self, init_obj, init_probe, init_crop_pos, init_probe_pos_shifts, H, lr_params=None, device='cuda:0'):
        super(PtychoAD_v2, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obj = cplx_from_np(init_obj, cplx_type="amp_phase", ndim=-1).to(self.device)
            self.opt_probe = cplx_from_np(init_probe, cplx_type="amp_phase", ndim=-1).to(self.device) 
            self.opt_probe_pos_shifts = torch.tensor(init_probe_pos_shifts, device=self.device)
            self.crop_pos = torch.tensor(init_crop_pos, dtype=torch.int32, device=self.device)
            self.H = torch.tensor(H, dtype=torch.complex64, device=self.device)
            self.roi_shape = init_probe.shape[-2:]
            self.shift_probes = (lr_params['probe_pos_shifts'] != 0) # Set shift_probes to False if lr_params['probe_pos_shifts'] = 0
            
            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obj': self.opt_obj,
                'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }

            self.optimizer_params = []
            if lr_params:
                for param_name, lr in lr_params.items():
                    if param_name in self.optimizable_tensors:
                        self.optimizable_tensors[param_name].requires_grad = (lr != 0)  # Set requires_grad based on learning rate
                        if lr != 0:
                            self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                    else:
                        print(f"Warning: '{param_name}' is not a valid parameter name.")

            print('PtychoAD major variables:')
            for name, tensor in self.optimizable_tensors.items():
                print(f"{name}: {tensor.shape}, {tensor.dtype}, device:{tensor.device}, grad:{tensor.requires_grad}, lr:{lr_params[name]:.0e}")
            
    def get_obj_ROI(self, indices):
        """ Get object ROI with integer coordinates """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        height, width  = self.roi_shape[0], self.roi_shape[1]
        object_patches = torch.zeros((len(indices), *self.opt_obj.shape[:2], height, width, 2)).to(self.device)

        for i, idx in enumerate(indices):
            height_start, height_end = self.crop_pos[idx,0], self.crop_pos[idx,0] + height
            width_start,  width_end  = self.crop_pos[idx,1], self.crop_pos[idx,1] + width
            object_patch      =  self.opt_obj[:, :, height_start:height_end, width_start:width_end, :] # object_patch (omode, D,H,W,2), 2 for the amp/phase channel
            object_patches[i] = object_patch

        return object_patches

    def get_probes(self, indices):
        """ Get probes for each position """
        # If you're not trying to optimize probe positions, there's not much point using sub-px shifted stationary probes
        # This function will return a single probe when self.shift_probes = False,
        # and would only be returning multiple sub-px shifted probes if you're optimizing self.opt_probe_pos_shifts

        if self.shift_probes:
            temp_probe = self.opt_probe.permute(0,3,1,2) # (pmode, Ny, Nx, 2) -> (pmode, 2, Ny, Nx)
            probes = torch.zeros((len(indices), *temp_probe.shape)).to(self.device) # (N, pmode, 2, Ny, Nx)

            for i, idx in enumerate(indices):
                tH = self.opt_probe_pos_shifts[idx][0] # Note that translate (a,b) is in unit of px, although the doc says it's fractional
                tW = self.opt_probe_pos_shifts[idx][1] # positive is moving to right/down for tW and tH.
                probes[i] = v2.functional.affine(temp_probe, translate = (tW, tH), interpolation=v2.InterpolationMode.BILINEAR, angle=0, scale=1, shear=0) 
            probes = probes.permute(0,1,3,4,2) # (N, pmode, Ny, Nx, 2)
        else:
            probes = self.opt_probe[None,...] # Extend a singleton N dimension, essentially using same probe for all samples
        
        return probes
        
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed as an array and representing the whole batch
        
        object_patches = self.get_obj_ROI(indices)
        probes = self.get_probes(indices)
        dp_fwd = multislice_forward_model_batch_all(object_patches, probes, self.H)
        
        return dp_fwd

# This is an archived (2024.03.16) implementation of processing the sample indices serially
# It's slower in multislice calculation, while the get_obj_ROI with STN is a bit faster than
# the batch processing version. Considering I might disable sub-px probe correction or replace 
# the STN approach entirely, I'll move on with the batch version for now.
# Note that the PtychoAD_serial is around 10 sec faster that PtychoAD_batch for 1 iter of 16384 CBEDs.
# The total time is ~ 3 min 30 sec for 1 iter when optimizing only the (1,8,592,592,2) object 
class PtychoAD_serial(torch.nn.Module):
    def __init__(self, init_obj, init_probe, init_crop_pos, H, lr_params=None, device='cuda:0'):
        super(PtychoAD_serial, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obj = cplx_from_np(init_obj, cplx_type="amp_phase", ndim=1).to(self.device)
            self.opt_probe = cplx_from_np(init_probe, cplx_type="amp_phase", ndim=-1).to(self.device)
            self.opt_crop_pos_shifts = torch.zeros(*init_crop_pos.shape, device=self.device)
            self.translate_matrices = prepare_stack_transform(init_crop_pos, init_obj.shape).to(self.device)
            self.H = torch.tensor(H, dtype=torch.complex64, device=self.device)
            self.roi_shape = init_probe.shape[-2:]
        
        # Create a dictionary to store the optimizable tensors
        self.optimizable_tensors = {
            'obj': self.opt_obj,
            'probe': self.opt_probe,
            'crop_pos_shifts': self.opt_crop_pos_shifts
        }

        self.optimizer_params = []
        if lr_params:
            for param_name, lr in lr_params.items():
                if param_name in self.optimizable_tensors:
                    self.optimizable_tensors[param_name].requires_grad = (lr != 0)  # Set requires_grad based on learning rate
                    if lr != 0:
                        self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                else:
                    print(f"Warning: '{param_name}' is not a valid parameter name.")

        print('PtychoAD major variables:')
        print('object         ', self.opt_obj.dtype, self.opt_obj.shape, self.opt_obj.device)
        print('probe          ', self.opt_probe.dtype, self.opt_probe.shape, self.opt_probe.device)
        print('crop_pos_shifts', self.opt_crop_pos_shifts.dtype, self.opt_crop_pos_shifts.shape, self.opt_crop_pos_shifts.device)

        print("\nCurrently optimizing with learning rates:")
        for param, lr in lr_params.items():
            print(f"{param}: {lr:.0e}")
    

    def get_obj_ROI(self, idx):
        """ Using affine transformation to get object ROI with sub-px shifts by resampling """
        # Quite frankly, doing an integer version of get_obj_ROI would be much faster
        # Although that would disable the probe position correction capability.
        
        # opt_obj.shape = (B,C,D,H,W) = (omode,2,D,H,W) here is the expected output shape for affine_grid
        # Expected cM = (B,3,4)
        
        height, width       = self.roi_shape[0], self.roi_shape[1]
        cM_shifts           = torch.zeros((self.opt_obj.shape[0],3,4)).to(self.device)
        cM_shifts[:,0,3]    = 2*self.opt_crop_pos_shifts[idx,1]/self.opt_obj.shape[-1] # tx, or W dimensiton
        cM_shifts[:,1,3]    = 2*self.opt_crop_pos_shifts[idx,0]/self.opt_obj.shape[-2] # ty, or H dimension
        cM                  = self.translate_matrices[idx] + cM_shifts
        
        grid                = torch.nn.functional.affine_grid(cM, self.opt_obj.size(), align_corners = False) # grid (omode,D,H,W,3), 3 corresponds to 3D transformation
        translated_obj      = torch.nn.functional.grid_sample(self.opt_obj, grid, align_corners=False)        # translated_obj (omode,2,D,H,W) or the same as opt_obj
        object_patch        =  translated_obj[:, :, :, :height,:width].permute(0,2,3,4,1)                     # object_patch (omode, D,H,W,2), 2 for the amp/phase channel
        return object_patch

    def forward(self, idx):
        """ Doing the forward pass and get an output diffraction pattern """
        object_patch = self.get_obj_ROI(idx)
        dp_fwd = multislice_forward_model_pmode_omode(object_patch, self.opt_probe, self.H)
        return dp_fwd

# This is an archived (2024.03.16) implementation of processing the sample indices in batch
# I'm saving this one while working on a version with integer get_obj_ROI. 
# This implementation shifts the object with affine transformation and is a bit slow.
class PtychoAD_batch(torch.nn.Module):
    def __init__(self, init_obj, init_probe, init_crop_pos, H, lr_params=None, device='cuda:0'):
        super(PtychoAD_batch, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obj = cplx_from_np(init_obj, cplx_type="amp_phase", ndim=1).to(self.device)
            self.opt_probe = cplx_from_np(init_probe, cplx_type="amp_phase", ndim=-1).to(self.device)
            self.opt_crop_pos_shifts = torch.zeros(*init_crop_pos.shape, device=self.device)
            self.translate_matrices = prepare_stack_transform(init_crop_pos, init_obj.shape).to(self.device)
            self.H = torch.tensor(H, dtype=torch.complex64, device=self.device)
            self.roi_shape = init_probe.shape[-2:]
        
        # Create a dictionary to store the optimizable tensors
        self.optimizable_tensors = {
            'obj': self.opt_obj,
            'probe': self.opt_probe,
            'crop_pos_shifts': self.opt_crop_pos_shifts
        }

        self.optimizer_params = []
        if lr_params:
            for param_name, lr in lr_params.items():
                if param_name in self.optimizable_tensors:
                    self.optimizable_tensors[param_name].requires_grad = (lr != 0)  # Set requires_grad based on learning rate
                    if lr != 0:
                        self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                else:
                    print(f"Warning: '{param_name}' is not a valid parameter name.")

        print('PtychoAD major variables:')
        print('object         ', self.opt_obj.dtype, self.opt_obj.shape, self.opt_obj.device)
        print('probe          ', self.opt_probe.dtype, self.opt_probe.shape, self.opt_probe.device)
        print('crop_pos_shifts', self.opt_crop_pos_shifts.dtype, self.opt_crop_pos_shifts.shape, self.opt_crop_pos_shifts.device)

        print("\nCurrently optimizing with learning rates:")
        for param, lr in lr_params.items():
            print(f"{param}: {lr:.0e}")

    def get_obj_ROI(self, indices):
        """ Using affine transformation to get object ROI with sub-px shifts by resampling """
        # Quite frankly, doing an integer version of get_obj_ROI would be much faster
        # Although that would disable the probe position correction capability.
        # I'll keep it as is but if we find an external probe position routiine,
        # It's strongly recommended to do integer version of get_obj_ROI
        # A temporary fix would be choose a different get_obj_ROI approach depends on the user input
        
        # opt_obj.shape = (B,C,D,H,W) = (omode,2,D,H,W) here is the expected output shape for affine_grid
        # Expected cM = (B,3,4)
        # object_patches = (N,B,C,D,H,W), N is the additional sample index within the input batch, B is now used for omode.
        
        height, width       = self.roi_shape[0], self.roi_shape[1]
        object_patches = torch.zeros((len(indices), *self.opt_obj.shape[:3], height, width)).to(self.device)

        for i, idx in enumerate(indices):
            
            cM_shifts           = torch.zeros((self.opt_obj.shape[0],3,4)).to(self.device)
            cM_shifts[:,0,3]    = 2*self.opt_crop_pos_shifts[idx,1]/self.opt_obj.shape[-1] # tx, or W dimensiton
            cM_shifts[:,1,3]    = 2*self.opt_crop_pos_shifts[idx,0]/self.opt_obj.shape[-2] # ty, or H dimension
            cM                  = self.translate_matrices[idx] + cM_shifts
            
            grid                = torch.nn.functional.affine_grid(cM, self.opt_obj.size(), align_corners = False) # grid (omode,D,H,W,3), 3 corresponds to 3D transformation
            translated_obj      = torch.nn.functional.grid_sample(self.opt_obj, grid, align_corners=False)        # translated_obj (omode,2,D,H,W) or the same as opt_obj
            object_patch        =  translated_obj[:, :, :, :height,:width]                     # object_patch (omode, D,H,W,2), 2 for the amp/phase channel
            object_patches[i] = object_patch
        object_patches = object_patches.permute(0,1,3,4,5,2)
        return object_patches

    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed as an array and representing the whole batch
        object_patches = self.get_obj_ROI(indices)
        dp_fwd = multislice_forward_model_batch_pmode_omode(object_patches, self.opt_probe, self.H)
        return dp_fwd