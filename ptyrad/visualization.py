## Defining visualization functions

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_forward_pass(model, indices, dp_power, pass_fig = None):
    """ Plot the forward pass for the input torch model """
    # The input is expected to be torch object and the attributes are all torch tensors and will be converted to numpy
    
    # probes_int = (N_i, Ny, Nx), float32 np array
    # obj_ROI = (N_i, omode, Nz, Ny, Nx), float32 np array
    # For probe, only plot the intensity of incoherently summed mixed-state probe
    # For object, only plot the phase of the 1st object mode and sums over z-slices
    # The dp_power here is for visualization purpose, the actual loss function has its own param field
    
    with torch.no_grad():
        probes               = model.get_probes(indices)
        probes_int           = probes.abs().pow(2).sum(1)
        model_CBEDs, obj_ROI = model(indices)
        measured_CBEDs       = model.get_measurements(indices)
        
        probes_int     = probes_int.detach().cpu().numpy()
        obj_ROI        = obj_ROI.detach().cpu().numpy()
        model_CBEDs    = model_CBEDs.detach().cpu().numpy()
        measured_CBEDs = measured_CBEDs.detach().cpu().numpy()
        
    fig, axs = plt.subplots(len(indices), 4, figsize=(24, 5.5*len(indices)))
    
    for i, idx in enumerate(indices):

        im00 = axs[i,0].imshow(probes_int[i]) 
        axs[i,0].set_title(f"Model probe intensity iter{idx}")
        fig.colorbar(im00, shrink=0.6)
        
        im01 = axs[i,1].imshow(obj_ROI[i,0,:,:,:].sum(0))
        axs[i,1].set_title(f"Model object phase (zsum) iter{idx}")
        fig.colorbar(im01, shrink=0.6)

        im02 = axs[i,2].imshow((model_CBEDs[i]**dp_power))
        axs[i,2].set_title(f"Model CBED^{dp_power} iter{idx}")
        fig.colorbar(im02, shrink=0.6)
        
        im03 = axs[i,3].imshow((measured_CBEDs[i]**dp_power))
        axs[i,3].set_title(f"Data CBED^{dp_power} iter{idx}")
        fig.colorbar(im03, shrink=0.6)
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
        
def plot_scan_positions(pos, img=None, offset=None, dot_scale=0.001, show_arrow=True):
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
    
    plt.scatter(x = pos[:,1], y = pos[:,0], c=np.arange(len(pos)), s=dot_scale*np.arange(len(pos)))
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
    
def plot_summary(output_path, loss_iters, niter, init_variables, model, save_fig=None):
    
    # loss curves
    plt.figure(figsize=(8,6))
    plt.title("Loss value vs. iterations", fontsize=16)
    plt.plot(np.array(loss_iters)[:,0], np.array(loss_iters)[:,1], marker='o')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Loss values", fontsize=16)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path + f"/summary_{niter}_loss.png")
    
    # Final forward pass
    indices = np.random.randint(0, len(model.measurements),2)
    fig = plot_forward_pass(model, indices, 0.5, pass_fig=True)
    if save_fig:
        fig.savefig(output_path + f"/summary_{niter}_forward_pass.png")
    
    
    # Visualize the probe modes. 
    # This is for visualization so each mode has its own colorbar.
    # See the actual probe amplitude output for absolute scale visualizaiton
    init_probe = init_variables['probe']
    opt_probe = model.opt_probe.detach().cpu().numpy()
    fig, axs = plt.subplots(2, len(opt_probe), figsize=(len(opt_probe)*2.5, 6))
    plt.suptitle("Probe modes amplitude", fontsize=18)
    for i in range(len(opt_probe)):
        ax_init = axs[0, i]
        ax_init.set_title(f"Init probe {i}")
        im_init = ax_init.imshow(np.abs(init_probe[i]))
        ax_init.axis('off')
        plt.colorbar(im_init, ax=ax_init, shrink=0.6)

        ax_opt = axs[1, i]
        ax_opt.set_title(f"Opt probe {i} iter{niter}")
        im_opt = ax_opt.imshow(np.abs(opt_probe[i]))
        ax_opt.axis('off')
        plt.colorbar(im_opt, ax=ax_opt, shrink=0.6)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path + f"/summary_{niter}_probe_modes.png")