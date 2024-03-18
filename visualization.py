import matplotlib.pyplot as plt
import numpy as np

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