## Defining PtychoAD class for the optimization object

from forward_model import  multislice_forward_model_pmode_omode
from utils import cplx_from_np, prepare_stack_transform
import torch

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