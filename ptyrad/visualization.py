## Defining visualization functions

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
from numpy.fft import fftshift, ifftn
import torch
from ptyrad.utils import make_sigmoid_mask

def plot_sigmoid_mask(Npix, relative_radius, relative_width):
    mask = make_sigmoid_mask(Npix, relative_radius, relative_width).detach().cpu().numpy()
    fig, axs = plt.subplots(1,2, figsize=(13,6))
    fig.suptitle(f"Sigmoid mask with radius = {relative_radius}, width = {relative_width}", fontsize=18)
    im = axs[0].imshow(mask)
    axs[0].axhline(y=Npix//2, c='r', linestyle='--')
    axs[1].plot(mask[Npix//2], c='r')
    fig.colorbar(im, shrink=0.7)
    plt.show()

def plot_forward_pass(model, indices, dp_power, show_fig=True, pass_fig=False):
    """ Plot the forward pass for the input torch model """
    # The input is expected to be torch object and the attributes are all torch tensors and will be converted to numpy
    
    # probes_int = (N_i, Ny, Nx), float32 np array
    # obj_ROI = (N_i, omode, Nz, Ny, Nx) -> (N_i, Nz, Ny, Nx), float32 np array
    # For probe, only plot the intensity of incoherently summed mixed-state probe
    # For object, only plot the phase of the weighted sum object mode and sums over z-slices
    # The dp_power here is for visualization purpose, the actual loss function has its own param field
    
    with torch.no_grad():
        probes                   = model.get_probes(indices)
        probes_int               = probes.abs().pow(2).sum(1)
        model_CBEDs, obj_patches = model(indices)
        omode_occu               = model.omode_occu
        measured_CBEDs           = model.get_measurements(indices)
        
        probes_int     = probes_int.detach().cpu().numpy()
        obja_ROI        = (obj_patches[...,0] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        objp_ROI        = (obj_patches[...,1] * omode_occu[:,None,None,None]).sum(1).detach().cpu().numpy() # obj_ROI = (N_i, Nz,Ny,Nx)
        model_CBEDs    = model_CBEDs.detach().cpu().numpy()
        measured_CBEDs = measured_CBEDs.detach().cpu().numpy()
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(len(indices), 5, figsize=(24, 5*len(indices)))
    plt.suptitle("Forward pass", fontsize=24)
    
    for i, idx in enumerate(indices):
        # Looping over the N_i dimension
        im00 = axs[i,0].imshow(probes_int[i]) 
        axs[i,0].set_title(f"Probe intensity idx{idx}", fontsize=16)
        fig.colorbar(im00, shrink=0.6)

        im01 = axs[i,1].imshow(obja_ROI[i].prod(0))
        axs[i,1].set_title(f"Object amp. (osum, zprod) idx{idx}", fontsize=16)
        fig.colorbar(im01, shrink=0.6)
        
        im02 = axs[i,2].imshow(objp_ROI[i].sum(0))
        axs[i,2].set_title(f"Object phase (osum, zsum) idx{idx}", fontsize=16)
        fig.colorbar(im02, shrink=0.6)

        im03 = axs[i,3].imshow((model_CBEDs[i]**dp_power))
        axs[i,3].set_title(f"Model CBED^{dp_power} idx{idx}", fontsize=16)
        fig.colorbar(im03, shrink=0.6)
        
        im04 = axs[i,4].imshow((measured_CBEDs[i]**dp_power))
        axs[i,4].set_title(f"Data CBED^{dp_power} idx{idx}", fontsize=16)
        fig.colorbar(im04, shrink=0.6)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig
        
def plot_scan_positions(pos, init_pos=None, img=None, offset=None, figsize=(8,8), dot_scale=0.001, show_arrow=True, show_fig=True, pass_fig=False):
    """ Plot the scan positions given an array of (N,2) """
    # The array is expected to have shape (N,2)
    # Each row is rendered as (y, x), or equivalently (height, width)
    # The dots are plotted with asending size and color changes to represent the relative order
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig = plt.figure(figsize = figsize)
    ax = plt.gca() # There's only 1 ax for plt.figure(), and plt.title is an Axes-level attribute so I need to pass the Axes out because I like plt.title layout better
    plt.title("Scan positions", fontsize=16)
    
    if img is not None:
        plt.imshow(img)
        pos = np.array(pos) + np.array(offset)
        plt.gca().invert_yaxis()  # Pre-flip y-axis so the y-axis is image-like no matter what
    
    if init_pos is None:
        plt.scatter(x=pos[:,1], y=pos[:,0], c=np.arange(len(pos)), s=dot_scale*np.arange(len(pos)), label='Scan positions')
    else:
        plt.scatter(x=init_pos[:,1], y=init_pos[:,0], c='C0', s=dot_scale, label='Init scan positions')
        plt.scatter(x=pos[:,1],      y=pos[:,0],      c='C1', s=dot_scale, label='Opt scan positions')
        plt.ylim(init_pos[:,0].min()-10, init_pos[:,0].max()+10)
        plt.xlim(init_pos[:,1].min()-10, init_pos[:,1].max()+10)
    
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
    if show_fig:
        plt.show()
    if pass_fig:
        return fig, ax
    
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

def plot_pos_grouping(pos, batches, circle_diameter=False, diameter_type='90%', figsize=(16,8), dot_scale=1, pass_fig=False):
    
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
                ax.scatter(x=first_point[1], y=first_point[0], s=dot_scale, color='r')
                ax.add_artist(circle)
                
                # Add annotation for "90% probe intensity"
                annotation_text = f"{diameter_type} probe intensity"
                annotation_x = first_point[1]
                annotation_y = first_point[0] #+ circle_diameter / 2 + 10  # Adjust the vertical offset as needed
                ax.annotate(annotation_text, xy=(annotation_x-circle_diameter/2, annotation_y-circle_diameter/2-3))
            
        ax.set_xlabel('X (obj coord, px)')
        ax.set_ylabel('Y (obj coord, px)')
        ax.set_xlim(pos[:,1].min()-10, pos[:,1].max()+10) # Show the full range to better visualize if a sub-group (like 'center') is selected
        ax.set_ylim(pos[:,0].min()-10, pos[:,0].max()+10)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    if pass_fig:
        return fig
    
def plot_loss_curves(loss_iters, last_n_iters=10, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    data = np.array(loss_iters)

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plot all loss values
    axs.plot(data[:,0], data[:,1], marker='o')

    # Plot the last n iters as an inset
    if len(data) > 20 and last_n_iters is not None:
        # Create inset subplot for zoomed-in plot
        axins = axs.inset_axes([0.45, 0.3, 0.4, 0.5])
        axins.plot(data[-last_n_iters:,0], data[-last_n_iters:,1], marker='o')
        axins.set_xlabel('Iterations', fontsize=12)
        axins.set_ylabel('Loss value', fontsize=12)
        axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        axs.indicate_inset_zoom(axins, edgecolor="gray")
        axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Loss value', fontsize=16)
    axs.set_title(f'Loss value: {data[-1,1]:.5f} at iter {int(data[-1,0])}', fontsize=16)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_probe_modes(init_probe, opt_probe, amp_or_phase='amplitude', real_or_fourier='real', phase_cmap=None, amplitude_cmap=None, show_fig=True, pass_fig=False):
    # The input probes are expected to be numpy array
    # This is for visualization so each mode has its own colorbar.
    # See the actual probe amplitude output for absolute scale visualizaiton
    
    # Get the power distribution
    # Although init_probe_int wouldn't change, calculate it takes < 0.2 ms so realy no need to pre-calculate it
    init_probe_int = np.abs(init_probe)**2 
    opt_probe_int = np.abs(opt_probe)**2
    init_probe_pow = np.sum(init_probe_int, axis=(-2,-1))/np.sum(init_probe_int) 
    opt_probe_pow = np.sum(opt_probe_int, axis=(-2,-1))/np.sum(opt_probe_int)
    
    if real_or_fourier == 'fourier':
    # While it might seem redundant, the sandwitch fftshift(fft(fftshift(probe)))) is needed for the following reason:
    # Although probe_fourier = fftn(fftshift(probe)) and probe_fourier = fftn(probe) gives the same abs(probe_fourier),
    # pre-fftshifting the probe back to corner gives more accurate phase angle while plotting the angle(probe_fourier)
    # On the other hand, fftn(probe) would generate additional phase shifts that looks like checkerboard artifact in angle(probe_fourier)
    # fftshift and ifftshift behaves the same when N is even, and there's no scaling for fftshift so I'll stick with fftshift for simplicity
        init_probe = fftshift(ifftn(fftshift(init_probe, axes=(-2,-1)), axes=(-2, -1), norm='ortho'), axes=(-2,-1))
        opt_probe  = fftshift(ifftn(fftshift(opt_probe, axes=(-2,-1)),  axes=(-2, -1), norm='ortho'), axes=(-2,-1))
    elif real_or_fourier =='real':
        pass
    else:
        raise ValueError("Please use 'real' or 'fourier' for probe mode visualization!")
        
    if amp_or_phase == 'phase':
        # Negative sign for consistency with chi(k), because psi = exp(-i*chi(k)). Overfocus should give positive phase shift near the edge of aperture
        # Scale the plotted phase by the amplitude so we can focus more on the relevant phases
        init_probe = -np.angle(init_probe)*np.abs(init_probe) 
        opt_probe  = -np.angle(opt_probe)*np.abs(opt_probe)
        cmap = phase_cmap if phase_cmap else 'twilight'
    elif amp_or_phase == 'amplitude' or amp_or_phase == 'amp':
        init_probe = np.abs(init_probe)
        opt_probe  = np.abs(opt_probe)
        cmap = amplitude_cmap if amplitude_cmap else 'viridis'
    else:
        raise ValueError("Please use 'amplitude' or 'phase' for probe mode visualization!")
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(2, len(opt_probe), figsize=(len(opt_probe)*2.5, 6))
    axs = axs[:,None] if len(opt_probe) == 1 else axs # Expand axs to (2,1) to handle the special case of 1 probe mode
    plt.suptitle(f"Probe modes {amp_or_phase} in {real_or_fourier} space", fontsize=18)
                
    for i in range(len(opt_probe)):
        ax_init = axs[0, i]
        ax_init.set_title(f"Init pmode {i}: {init_probe_pow[i]:.1%}")
        im_init = ax_init.imshow(init_probe[i], cmap=cmap)
        ax_init.axis('off')
        plt.colorbar(im_init, ax=ax_init, shrink=0.6)

        ax_opt = axs[1, i]
        ax_opt.set_title(f"Opt pmode {i}: {opt_probe_pow[i]:.1%}")
        im_opt = ax_opt.imshow(opt_probe[i], cmap=cmap)
        ax_opt.axis('off')
        plt.colorbar(im_opt, ax=ax_opt, shrink=0.6)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_summary(output_path, loss_iters, niter, indices, init_variables, model, show_fig=True, save_fig=False):
    """ Wrapper function for most visualization function """
    # Note: Set show_fig=False and save_fig=True if you just want to save the figure without showing
    
    # Sets figure saving to be True if you accidiently disable both show_fig and save_fig
    if show_fig is False and save_fig is False:
        save_fig = True 
    
    # Forward pass
    fig_forward = plot_forward_pass(model, np.random.choice(indices,2, replace=False), 0.5, show_fig=False, pass_fig=True)
    fig_forward.suptitle(f"Forward pass at iter {niter}", fontsize=24)
  
    # loss curves
    fig_loss = plot_loss_curves(loss_iters, last_n_iters=10, show_fig=show_fig, pass_fig=True)
    
    # Probe modes in real and reciprocal space
    init_probe = init_variables['probe']
    opt_probe = model.opt_probe.detach().cpu().numpy()
    fig_probe_modes_real_amp      = plot_probe_modes(init_probe, opt_probe, real_or_fourier='real',    amp_or_phase='amplitude', show_fig=False, pass_fig=True)
    fig_probe_modes_fourier_amp   = plot_probe_modes(init_probe, opt_probe, real_or_fourier='fourier', amp_or_phase='amplitude', show_fig=False, pass_fig=True)
    fig_probe_modes_fourier_phase = plot_probe_modes(init_probe, opt_probe, real_or_fourier='fourier', amp_or_phase='phase', show_fig=False, pass_fig=True)
    fig_probe_modes_real_amp.suptitle(f"Probe modes amplitude in real space at iter {niter}", fontsize=18)
    fig_probe_modes_fourier_amp.suptitle(f"Probe modes amplitude in fourier space at iter {niter}", fontsize=18)
    fig_probe_modes_fourier_phase.suptitle(f"Probe modes phase in fourier space at iter {niter}", fontsize=18)

    # Scan positions
    init_pos = init_variables['crop_pos'] + init_variables['probe_pos_shifts']
    pos = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()
    fig_scan_pos, ax = plot_scan_positions(pos=pos[indices], init_pos=init_pos[indices], dot_scale=1, show_fig=False, pass_fig=True)
    ax.set_title(f"Scan positions at iter {niter}", fontsize=16)
        
    # Show and save fig
    if show_fig:
        fig_loss.show()
        fig_forward.show()
        fig_probe_modes_real_amp.show()
        fig_probe_modes_fourier_amp.show()
        fig_probe_modes_fourier_phase.show()
        fig_scan_pos.show()

    if save_fig:
        print(f"Saving summary figures for iter {niter}")
        fig_loss.savefig(output_path + f"/summary_loss_iter{str(niter).zfill(4)}.png")
        fig_forward.savefig(output_path + f"/summary_forward_pass_iter{str(niter).zfill(4)}.png")
        fig_probe_modes_real_amp.savefig(output_path + f"/summary_probe_modes_real_amp_iter{str(niter).zfill(4)}.png",bbox_inches='tight')
        fig_probe_modes_fourier_amp.savefig(output_path + f"/summary_probe_modes_fourier_amp_iter{str(niter).zfill(4)}.png",bbox_inches='tight')
        fig_probe_modes_fourier_phase.savefig(output_path + f"/summary_probe_modes_fourier_phase_iter{str(niter).zfill(4)}.png",bbox_inches='tight')
        fig_scan_pos.savefig(output_path + f"/summary_scan_pos_iter{str(niter).zfill(4)}.png")
        
    # Close figures after saving
    plt.close(fig_loss)
    plt.close(fig_forward)
    plt.close(fig_probe_modes_real_amp)
    plt.close(fig_probe_modes_fourier_amp)
    plt.close(fig_probe_modes_fourier_phase)
    plt.close(fig_scan_pos)