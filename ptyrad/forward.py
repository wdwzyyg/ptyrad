## Defining multislice forward model for electron diffraction with mixed probe/object modes of 3D objects

from torch.fft import fft2, ifft2
from .utils import fftshift2
import torch

# This is a current working version (2024.04.03) of the multislice forward model
# I cleaned up the archived versiosn and slightly renamed the objects and variables for clarity

# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully, 
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster 
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing

def multislice_forward_model_vec_all(object, omode_occu, probe, H):
    """ multislice electron diffraction with multiple incoherent probe and object modes """
    # Example usage:
    # dp_fwd = multislice_forward_model_vec_all(object_patches, omode_occu, probes, H)

    # object (N, omode, Nz, Ny, Nx, 2), pseudo-complex tensor with float32. N for samples in a batch
    # probe (N, pmode, Ny, Nx), tensor with complex64. Default probe N is 1, as using the same probe for all samples when there's no sub-px probe shifts
    # H (Ky, Kx), tensor with complex64. This is the Fresnel propagator that propagates the wave by a slice thickness
    # omode_occu (omode), tensor with float32. This determines the occupancy/expectation for each object mode, sum(omode_occu) = 1
    # dp_fwd (N, Ky, Kx), tensor with float32
    
    # Cast the object back to actual complex tensor
    object_cplx = torch.polar(object[...,0], object[...,1]) # (N, omode, Nz, Ny, Nx)
    n_slices = object_cplx.shape[2]
    
    # Expand psi to include omode dimension
    psi = probe[:, :, None, :, :] # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices-1):
        object_slice = object_cplx[:, :, n, :, :] # object_slice -> (N, omode, Ny, Nx)
        psi = psi * object_slice[:, None, :, :, :]  # psi -> (N, pmode, omode, Ny, Nx)
        psi = ifft2(H * fft2(psi)) # Note that fft2 and ifft2 are applying to the last 2 axis

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[:, :, n_slices-1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from probe / object modes are incoherently summed together

    # Breaking down the steps for clarity, while combine all of these for lower peak memory consumption
    # psi_k = fftshift(fft2(psi, dim=(-2, -1)), dim=(-2, -1)
    # |psi_k|^2 = psi_k.abs().square()
    # weighted_psi_k = |psi_k|^2 * omode_occu
    # dp_fwd = sum(weighted_psi_k)
    # Note that norm = 'ortho' is needed to ensure the for each sample, sum(|psi|^2) and sum(dp) has the same scale (should be 1) 
    
    dp_fwd = torch.sum(torch.square(torch.abs(fftshift2(fft2(psi, norm='ortho')))) * omode_occu[:,None,None], dim=(1, 2)) + 1e-20 # Add 1e-20 for numerical stability
    return dp_fwd