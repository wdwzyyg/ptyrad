## Defining PtychoAD class for the optimization object

from forward import  multislice_forward_model_vec_all
from utils import imshift_batch
import torch

# This is a current working version (2024.03.23) of the PtychoAD class
# I cleaned up the archived versiosn and slightly renamed the objects and variables for clarity

# set_optimizer function is called at the end of the initializaiton, while this can also be called if you want to update the optimizer params without initializing the object
# obj optimization is now split into objp and obja
# mixed object modes are normalized by the init_omode_occu. By design this is a fixed value because optimizing omode_occu with obj simultaneously could be unstable
# obj_ROI cropping is done with vectorization and the obj_ROI_grid is only generated once
# probe with sub-px shifts are calculated only when probe_pos_shifts are enables for optimization
# All the sub-px shifted probes in a batch are processed together with vectorizaiton
# Likewise, the multislice forward model is also fully vectorized across samples (in batch), pmode, and omode
# Note that it's possible to reduce the peak-memory consumption by reducing the level of vectorizaiton and roll back to for loops
# Lastly, the forward pass of this model would output the dp_fwd (N, Ky, Kx) and objp_patches (N, omode, Nz, Ny, Nx) in float32 for later loss calculation

class PtychoAD(torch.nn.Module):
    """ Main optimization class for the ptycho reconstruction using AD """
    # Including initialization, get_obj_ROI, get_probes, and forward methods
    
    # Example usage:
    # model = PtychoAD(init_obj, init_omode_occu, init_probe, init_crop_pos, init_probe_pos_shifts, H, 
    #                  lr_params={'obja': 0,
    #                             'objp': 3e-4,
    #                             'probe': 1e-5, 
    #                             'probe_pos_shifts': 0},
    #                  device=DEVICE)
    # optimizer = torch.optim.Adam(model.optimizer_params)

    def __init__(self, init_obj, init_omode_occu, init_probe, init_crop_pos, init_probe_pos_shifts, H, lr_params=None, device='cuda:0'):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            self.device = device
            self.opt_obja  = torch.abs(torch.tensor(init_obj, dtype=torch.complex64, device=device))
            self.opt_objp  = torch.angle(torch.tensor(init_obj, dtype=torch.complex64, device=device))
            self.opt_probe = torch.tensor(init_probe, dtype=torch.complex64, device=device)  
            self.opt_probe_pos_shifts = torch.tensor(init_probe_pos_shifts, device=device)
            self.omode_occu = torch.tensor(init_omode_occu, dtype=torch.float32, device=device) 
            self.H = torch.tensor(H, dtype=torch.complex64, device=device)
            self.shift_probes = (lr_params['probe_pos_shifts'] != 0) # Set shift_probes to False if lr_params['probe_pos_shifts'] = 0
            
            Ny, Nx = init_probe.shape[-2:]
            ry, rx = torch.meshgrid(torch.arange(Ny, dtype=torch.int32, device=device), torch.arange(Nx, dtype=torch.int32, device=device), indexing='ij')
            self.shift_probes_grid = torch.stack([ry/Ny, rx/Nx], dim=0)
            # Create the grid for obj_ROI in a vectorized approach
            # ry is the y-grid (Ny,Nx), by adding the y coordinates from init_crop_pos (N,1) in a broadcast way, it becomes (N,Ny,Nx)
            # Stacking the modified ry and rx at the last dimension, we get obj_ROI_grid = (N,Ny,Nx,2)
            self.obj_ROI_grid = torch.stack([ry[None,:,:] + torch.tensor(init_crop_pos[:, None, None, 0], device=device), 
                                             rx[None,:,:] + torch.tensor(init_crop_pos[:, None, None, 1], device=device)], dim=-1)
            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obja'            : self.opt_obja,
                'objp'            : self.opt_objp,
                'probe'           : self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts}
            self.set_optimizer(lr_params)
    
    def set_optimizer(self, lr_params):
        self.optimizer_params = []
        if lr_params:
            for param_name, lr in lr_params.items():
                if param_name in self.optimizable_tensors:
                    self.optimizable_tensors[param_name].requires_grad = (lr != 0) # Set requires_grad based on learning rate
                    if lr != 0:
                        self.optimizer_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
                else:
                    print(f"Warning: '{param_name}' is not a valid parameter name.")
        # Declaring it as a ParameterDict so that I can use model.state_dict()
        # Note that when I wrap the former dict directly with ParameterDict it disables their grad_fn for unknown reason
        self.nn_params = torch.nn.ParameterDict(self.optimizable_tensors)
        
        print('PtychoAD optimizable variables:')
        for name, tensor in self.optimizable_tensors.items():
            print(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{lr_params[name]:.0e}")
        print('\nMake sure to pass the optimizer_params to optimizer object using "optimizer = torch.optim.Adam(model.optimizer_params)"')
    
    def get_obj_ROI(self, indices):
        """ Get object ROI with integer coordinates """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        object_patches = opt_obj[:,:,self.obj_ROI_grid[indices,...,0],self.obj_ROI_grid[indices,...,1],:].permute(2,0,1,3,4,5)
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
        # The indices are passed in as an array and representing the whole batch
        
        object_patches = self.get_obj_ROI(indices)
        probes = self.get_probes(indices)
        dp_fwd = multislice_forward_model_vec_all(object_patches, self.omode_occu, probes, self.H)
        return dp_fwd, object_patches[...,1]