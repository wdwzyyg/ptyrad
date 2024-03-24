## Defining visualization functions

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_forward_pass(model, indices, dp_power, object_data, crop_indices_data, cbeds_data):
    with torch.no_grad():
        dp_fwd, obj_ROI = model(indices)
        height, width = model.opt_probe.shape[-2], model.opt_probe.shape[-1]

    for i, idx in enumerate(indices):

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        # object_data (omode, Nz, Ny, Nx), complex64 np array
        # obj_ROI = (N, omode, Nz, Ny, Nx)
        # currently only plotting the phase of the 1st object mode and sums over z-slices
        
        object_patch = np.angle(object_data[0,:,crop_indices_data[idx,0]:crop_indices_data[idx,0]+height,crop_indices_data[idx,1]:crop_indices_data[idx,1]+width]).sum(0)
        
        im00 = axs[0, 0].imshow(obj_ROI[i,0,:,:,:].sum(0).detach().cpu()) 
        axs[0, 0].set_title(f"Model obj {idx}")
        fig.colorbar(im00)
        
        im01 = axs[0, 1].imshow(object_patch)
        axs[0, 1].set_title(f"Data obj {idx}")
        fig.colorbar(im01)

        im10 = axs[1, 0].imshow((dp_fwd[i]**dp_power).detach().cpu())
        axs[1, 0].set_title(f"Model CBED {idx}")
        fig.colorbar(im10)
        
        im11 = axs[1, 1].imshow(cbeds_data[idx]**dp_power)
        axs[1, 1].set_title(f"Data CBED {idx}")
        fig.colorbar(im11)
        plt.show()