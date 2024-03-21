import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_forward_pass(model, indices, dp_power, object_data, crop_indices_data, cbeds_data):
    with torch.no_grad():
        dp_fwd = model(indices).detach().cpu()
        obj_ROI = model.get_obj_ROI(indices).detach().cpu()
        height, width = model.opt_probe.shape[-2], model.opt_probe.shape[-2]

    for i, idx in enumerate(indices):

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        object_patch = np.angle(object_data[0,0,crop_indices_data[idx,0]:crop_indices_data[idx,0]+height,crop_indices_data[idx,1]:crop_indices_data[idx,1]+width])
        
        axs[0, 0].imshow(obj_ROI[i,0,:,:,:,1].sum(0)) # Only plotting the 1st object mode (N, omode, Nz, Ny, Nx, C)
        axs[0, 0].set_title(f"Model obj {idx}")

        axs[0, 1].imshow(object_patch)
        axs[0, 1].set_title(f"Data obj {idx}")

        im0 = axs[1, 0].imshow(dp_fwd[i]**dp_power)
        axs[1, 0].set_title(f"Model CBED {idx}")
        fig.colorbar(im0)
        
        im1 = axs[1, 1].imshow(cbeds_data[idx]**dp_power)
        axs[1, 1].set_title(f"Data CBED {idx}")
        fig.colorbar(im1)
        plt.show()
        
# This one would need some work
def plot_recon_progress(iter, batch_num, AD_image, Input_image):
    # Plot after each update
    fig, axs = plt.subplots(1, 4, figsize=(28, 6))
    im3 = None  # Initialize the image object for the input image
        
    fig.suptitle(f"Batch {batch_num} in iter {iter}", fontsize=16)
    
    # Determine the common display range
    vmin, vmax = np.min(Input_image), np.max(Input_image)
    
    # Plot the first three images
    for i, ax in enumerate(axs[:3]):
        ax.imshow(AD_image[i], cmap='viridis',  vmin=vmin, vmax=vmax)
        ax.set_title(f'AD image mode {i}')

    # Plot the input image
    if im3 is None:
        im3 = axs[3].imshow(Input_image, cmap='viridis',  vmin=vmin, vmax=vmax)
    else:
        im3.set_data(Input_image)

    axs[3].set_title('Input image')

    # Update the colorbar
    fig.colorbar(im3, ax=axs, orientation='vertical', shrink=0.6)

    return fig