## Defining visualization functions

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftshift, fftn
import torch

def plot_forward_pass(model, indices, dp_power, pass_fig=False):
    """ Plot the forward pass for the input torch model """
    # The input is expected to be torch object and the attributes are all torch tensors and will be converted to numpy
    
    # probes_int = (N_i, Ny, Nx), float32 np array
    # obj_ROI = (N_i, omode, Nz, Ny, Nx) -> (N_i, Nz, Ny, Nx), float32 np array
    # For probe, only plot the intensity of incoherently summed mixed-state probe
    # For object, only plot the phase of the weighted sum object mode and sums over z-slices
    # The dp_power here is for visualization purpose, the actual loss function has its own param field
    
    with torch.no_grad():
        probes               = model.get_probes(indices)
        probes_int           = probes.abs().pow(2).sum(1)
        model_CBEDs, obj_ROI = model(indices)
        omode_occu           = model.omode_occu
        measured_CBEDs       = model.get_measurements(indices)
        
        
        probes_int     = probes_int.detach().cpu().numpy()
        obj_ROI        = (obj_ROI * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        model_CBEDs    = model_CBEDs.detach().cpu().numpy()
        measured_CBEDs = measured_CBEDs.detach().cpu().numpy()
        
    fig, axs = plt.subplots(len(indices), 4, figsize=(24, 5.5*len(indices)))
    
    for i, idx in enumerate(indices):

        im00 = axs[i,0].imshow(probes_int[i]) 
        axs[i,0].set_title(f"Model probe intensity index{idx}", fontsize=16)
        fig.colorbar(im00, shrink=0.6)
        
        im01 = axs[i,1].imshow(obj_ROI[i].sum(0))
        axs[i,1].set_title(f"Model object phase (osum, zsum) index{idx}", fontsize=16)
        fig.colorbar(im01, shrink=0.6)

        im02 = axs[i,2].imshow((model_CBEDs[i]**dp_power))
        axs[i,2].set_title(f"Model CBED^{dp_power} index{idx}", fontsize=16)
        fig.colorbar(im02, shrink=0.6)
        
        im03 = axs[i,3].imshow((measured_CBEDs[i]**dp_power))
        axs[i,3].set_title(f"Data CBED^{dp_power} index{idx}", fontsize=16)
        fig.colorbar(im03, shrink=0.6)
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
        
def plot_scan_positions(pos, init_pos=None, img=None, offset=None, figsize=(8,8), dot_scale=0.001, show_arrow=True, pass_fig=False):
    """ Plot the scan positions given an array of (N,2) """
    # The array is expected to have shape (N,2)
    # Each row is rendered as (y, x), or equivalently (height, width)
    # The dots are plotted with asending size and color changes to represent the relative order
    
    fig = plt.figure(figsize = figsize)
    plt.title("Scan positions")
    
    if img is not None:
        plt.imshow(img)
        pos = np.array(pos) + np.array(offset)
        plt.gca().invert_yaxis()  # Pre-flip y-axis so the y-axis is image-like no matter what
    
    if init_pos is None:
        plt.scatter(x=pos[:,1], y=pos[:,0], c=np.arange(len(pos)), s=dot_scale*np.arange(len(pos)), label='Scan positions')
    else:
        plt.scatter(x=init_pos[:,1], y=init_pos[:,0], c='C0', s=dot_scale, label='Init scan positions')
        plt.scatter(x=pos[:,1],      y=pos[:,0],      c='C1', s=dot_scale, label='Opt scan positions')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
        
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    # Draw arrow from 1st position to 10th position
    if show_arrow:
        plt.arrow(pos[0, 1], pos[0, 0], pos[9, 1] - pos[0, 1], pos[9, 0] - pos[0, 0],
                color='red', head_width=2.5, head_length=5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
    
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

def plot_pos_grouping(pos, batches, circle_diameter=False, figsize=(16,8), dot_scale=1, pass_fig=False):
    
    fig, axs = plt.subplots(1,2, figsize = figsize)
    
    for i, ax in enumerate(axs):
        if i == 0:
            axs[0].set_title("Scan positions for all groups", fontsize=18)
            for batch in batches:
                ax.scatter(x=pos[batch, 1], y=pos[batch, 0], s=dot_scale)
        else:
            axs[1].set_title("Scan positions from group 0", fontsize=18)
            ax.scatter(x=pos[batches[0], 1], y=pos[batches[0], 0], s=dot_scale)
            
            # Draw a circle at the first point with the given diameter
            if circle_diameter:
                first_point = pos[batches[0][0]]
                circle = plt.Circle((first_point[1], first_point[0]), circle_diameter / 2, fill=False, color='r', linestyle='--')
                ax.add_artist(circle)
                
                # Add annotation for "90% probe intensity"
                annotation_text = "90% probe intensity"
                annotation_x = first_point[1]
                annotation_y = first_point[0] #+ circle_diameter / 2 + 10  # Adjust the vertical offset as needed
                ax.annotate(annotation_text, xy=(annotation_x-circle_diameter/2, annotation_y-circle_diameter/2-3))
            
        ax.set_xlabel('X (obj coord, px)')
        ax.set_ylabel('Y (obj coord, px)')
        ax.set_xlim(pos[:,1].min()-10, pos[:,1].max()+10)
        ax.set_ylim(pos[:,0].min()-10, pos[:,0].max()+10)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
    
def plot_loss_curves(loss_iters, pass_fig=False):
    fig = plt.figure(figsize=(8,6))
    plt.title("Loss value vs. iterations", fontsize=16)
    plt.plot(np.array(loss_iters)[:,0], np.array(loss_iters)[:,1], marker='o')
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Loss values", fontsize=16)
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
    
def plot_probe_modes(init_probe, opt_probe, amp_or_phase='amplitude', real_or_fourier='real', pass_fig=False):
    # This is for visualization so each mode has its own colorbar.
    # See the actual probe amplitude output for absolute scale visualizaiton
    
    if real_or_fourier == 'fourier':
        init_probe = fftshift(fftn(init_probe, axes=(-2, -1), norm='ortho'), axes=(-2, -1))
        opt_probe  = fftshift(fftn(opt_probe,  axes=(-2, -1), norm='ortho'), axes=(-2, -1))
    elif real_or_fourier =='real':
        pass
    else:
        raise ValueError("Please use 'real' or 'fourier' for probe mode visualization!")
        
    if amp_or_phase == 'phase':
        init_probe = np.angle(init_probe)
        opt_probe  = np.angle(opt_probe)
    elif amp_or_phase == 'amplitude' or amp_or_phase == 'amp':
        init_probe = np.abs(init_probe)
        opt_probe  = np.abs(opt_probe)
    else:
        raise ValueError("Please use 'amplitude' or 'phase' for probe mode visualization!")
    
    fig, axs = plt.subplots(2, len(opt_probe), figsize=(len(opt_probe)*2.5, 6))
    axs = axs[:,None] if len(opt_probe) == 1 else axs # Expand axs to (2,1) to handle the special case of 1 probe mode
    plt.suptitle(f"Probe modes {amp_or_phase} in {real_or_fourier} space", fontsize=18)
                
    for i in range(len(opt_probe)):
        ax_init = axs[0, i]
        ax_init.set_title(f"Init probe mode {i}")
        im_init = ax_init.imshow(init_probe[i])
        ax_init.axis('off')
        plt.colorbar(im_init, ax=ax_init, shrink=0.6)

        ax_opt = axs[1, i]
        ax_opt.set_title(f"Opt probe mode {i}")
        im_opt = ax_opt.imshow(opt_probe[i])
        ax_opt.axis('off')
        plt.colorbar(im_opt, ax=ax_opt, shrink=0.6)
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig

def plot_summary(output_path, loss_iters, niter, indices, init_variables, model, save_fig=False):
    """ Wrapper function for most visualization function """
    
    # Forward pass
    fig_forward = plot_forward_pass(model, np.random.choice(indices,2, replace=False), 0.5, pass_fig=True)
    if save_fig:
        fig_forward.savefig(output_path + f"/summary_forward_pass_iter{str(niter).zfill(4)}.png")
    
    # loss curves
    fig_loss = plot_loss_curves(loss_iters, pass_fig=True)
    if save_fig:
        fig_loss.savefig(output_path + f"/summary_loss_iter{str(niter).zfill(4)}.png")
    
    # Probe modes in real and reciprocal space
    fig_probe_modes_real = plot_probe_modes(init_variables['probe'], model.opt_probe.detach().cpu().numpy(), real_or_fourier='real', pass_fig=True)
    fig_probe_modes_fourier = plot_probe_modes(init_variables['probe'], model.opt_probe.detach().cpu().numpy(), real_or_fourier='fourier', pass_fig=True)
    if save_fig:
        fig_probe_modes_real.savefig(output_path + f"/summary_probe_modes_real_iter{str(niter).zfill(4)}.png",bbox_inches='tight')
        fig_probe_modes_fourier.savefig(output_path + f"/summary_probe_modes_fourier_iter{str(niter).zfill(4)}.png",bbox_inches='tight')
        
    # Scan positions
    init_pos = init_variables['crop_pos'] + init_variables['probe_pos_shifts']
    pos = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    fig_scan_pos = plot_scan_positions(pos=pos, init_pos=init_pos, dot_scale=1, pass_fig=True)
    if save_fig:
        fig_scan_pos.savefig(output_path + f"/summary_scan_pos_iter{str(niter).zfill(4)}.png")