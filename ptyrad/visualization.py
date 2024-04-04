## Defining visualization functions

import matplotlib.pyplot as plt
import numpy as np
import torch

# I can probably vectorize the obj cropping part,
# but honestly once we start from scratch without ground truth/previous run
# I guess this function would need some major change.

# Initially this function seems to be used to check whether my model implementaiton is working independently correct
# But after some reduction, it now become depend to the model, mmm

def plot_forward_pass(model, indices, dp_power, object_data):
    with torch.no_grad():
        model_CBEDs, obj_ROI = model(indices)
        measured_CBEDs = model.get_measurements(indices)
        height, width = model.opt_probe.shape[-2], model.opt_probe.shape[-1]
        crop_indices = model.crop_pos
        
    for i, idx in enumerate(indices):

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # object_data (omode, Nz, Ny, Nx), complex64 np array
        # obj_ROI = (N, omode, Nz, Ny, Nx)
        # currently only plotting the phase of the 1st object mode and sums over z-slices
        
        object_patch = np.angle(object_data[0,:,crop_indices[idx,0]:crop_indices[idx,0]+height,crop_indices[idx,1]:crop_indices[idx,1]+width]).sum(0)
        
        im00 = axs[0, 0].imshow(obj_ROI[i,0,:,:,:].sum(0).detach().cpu()) 
        axs[0, 0].set_title(f"Model obj {idx}")
        fig.colorbar(im00)
        
        im01 = axs[0, 1].imshow(object_patch)
        axs[0, 1].set_title(f"Data obj {idx}")
        fig.colorbar(im01)

        im10 = axs[1, 0].imshow((model_CBEDs[i]**dp_power).detach().cpu())
        axs[1, 0].set_title(f"Model CBED {idx}")
        fig.colorbar(im10)
        
        im11 = axs[1, 1].imshow((measured_CBEDs[i]**dp_power).detach().cpu())
        axs[1, 1].set_title(f"Data CBED {idx}")
        fig.colorbar(im11)
        plt.show()