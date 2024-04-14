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
        
def plot_scan_positions(pos, img=None, offset=None, show_arrow=True):
    """ Plot the scan positions given an array of (N,2) """
    # The array is expected to have shape (N,2)
    # Each row is rendered as (y, x), or equivalently (height, width)
    # The dots are plotted with asending size and color changes to represent the relative order
    
    plt.figure(figsize = (8,8))
    plt.title("Scan positions")
    
    if img is not None:
        plt.imshow(img)
        pos = np.array(pos) + np.array(offset)
        plt.gca().invert_yaxis()  # Pre-flip y-axis so the y-axis is image-like no matter what
    
    plt.scatter(x = pos[:,1], y = pos[:,0], c=np.arange(len(pos)), s=0.001*np.arange(len(pos)))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
        
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    # Draw arrow from 1st position to 10th position
    if show_arrow:
        plt.arrow(pos[0, 1], pos[0, 0], pos[9, 1] - pos[0, 1], pos[9, 0] - pos[0, 0],
                color='red', head_width=2.5, head_length=5)
    plt.show()
    
def plot_affine_transformation(scale, asymmetry, rotation, shear):
    from .utils import compose_affine_matrix
    # Example
    # plot_affine_transformation(2,0,45,0)
    A = np.eye(2)
    Af = compose_affine_matrix(scale, asymmetry, rotation, shear)
    
    plt.figure()
    plt.title("Visualize affine transformation")

    # Add origin and scatter points
    plt.scatter(0, 0, color='gray', marker='o', s=3)
    plt.scatter(A[:,1], A[:,0], label='Original')
    plt.scatter(Af[:,1], Af[:,0], label='Transformed')

    # Adding arrows
    plt.quiver(A[0,1], A[0,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(A[1,1], A[1,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(Af[0,1], Af[0,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)
    plt.quiver(Af[1,1], Af[1,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)

    # Adding grid lines
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.legend()
    plt.show()