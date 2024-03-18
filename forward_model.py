
import torch
from torch.fft import fft2, ifft2, fftshift

## Defining forward models

# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing

# The batch computation is much faster than the serial version, however, actual forward pass could be
# having bottleneck at the get_obj_ROI part and for some reason it's a bit slower to do get_obj_ROI in batch.


def multislice_forward_model_batch_all(object, probe, H):
    """ multislice electron diffraction with multiple incoherent probe and object modes"""
    # object (N, omode, Nz, Ny, Nx, 2), pseudo-complex tensor with float32. N for samples in a batch
    # probe (N, pmode, Ny, Nx, 2), tensor with float32. Default probe N is 1, as using the same probe for all samples
    # H (Ky, Kx), tensor with complex64
    # dp_fwd (B, Ky, Kx), tensor with float32
    
    # Cast the object/probe back to actual complex tensor
    object_cplx = torch.polar(object[...,0], object[...,1]) # (N, omode, Nz, Ny, Nx)
    probe_cplx = torch.polar(probe[...,0], probe[...,1])    # (pmode, Ny, Nx)
    n_slices = object_cplx.shape[2]
    
    # Expand psi to include omode dimension
    psi = probe_cplx[:, :, None, :, :] # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices-1):
        object_slice = object_cplx[:, :, n, :, :] # object_slice -> (N, omode, Ny, Nx)
        psi = psi * object_slice[:, None, :, :, :]  # psi -> (N, pmode, omode, Ny, Nx)
        psi = ifft2(H * fft2(psi, dim=(-2, -1)), dim=(-2, -1))

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[:, :, n_slices-1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from incoherent probe / object modes are summed together
    dp_fwd = torch.sum(torch.square(torch.abs(fftshift(fft2(psi, dim=(-2, -1)), dim=(-2, -1)))), dim=(1, 2))
    return dp_fwd

# Batch object version of multislice forward model
def multislice_forward_model_batch_pmode_omode(object, probe, H):
    """ multislice electron diffraction with multiple incoherent probe and object modes"""
    # object (N, omode, Nz, Ny, Nx, 2), pseudo-complex tensor with float32. N for samples in a batch
    # probe (pmode, Ny, Nx, 2), tensor with float32
    # H (Ky, Kx), tensor with complex64
    # dp_fwd (B, Ky, Kx), tensor with float32
    
    # Cast the object/probe back to actual complex tensor
    object_cplx = torch.polar(object[...,0], object[...,1]) # (N, omode, Nz, Ny, Nx)
    probe_cplx = torch.polar(probe[...,0], probe[...,1])    # (pmode, Ny, Nx)
    n_slices = object_cplx.shape[2]
    
    # Expand psi to includ sample and omode dimension
    psi = probe_cplx[None, :, None, :, :] # (pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices-1):
        object_slice = object_cplx[:, :, n, :, :] # object_slice -> (N, omode, Ny, Nx)
        psi = psi * object_slice[:, None, :, :, :]  # psi -> (N, pmode, omode, Ny, Nx)
        psi = ifft2(H * fft2(psi, dim=(-2, -1)), dim=(-2, -1))

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[:, :, n_slices-1, :, :]
    psi = psi * object_slice[:, None, :, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from incoherent probe / object modes are summed together
    dp_fwd = torch.sum(torch.square(torch.abs(fftshift(fft2(psi, dim=(-2, -1)), dim=(-2, -1)))), dim=(1, 2))
    return dp_fwd

# Serial version of multislice forward model, this is archived on 2024.03.16
def multislice_forward_model_pmode_omode(object, probe, H):
    """ multislice electron diffraction with multiple incoherent probe and object modes"""
    # object (omode, Nz, Ny, Nx, 2), pseudo-complex tensor with float32
    # probe (pmode, Ny, Nx, 2), tensor with float32
    # H (Ky, Kx), tensor with complex64
    # dp_fwd (Ky, Kx), tensor with float32
    
    # Cast the object/probe back to actual complex tensor
    object_cplx = torch.polar(object[...,0], object[...,1]) # (omode, Nz, Ny, Nx)
    probe_cplx = torch.polar(probe[...,0], probe[...,1]) # (pmode, Ny, Nx)
    n_slices = object_cplx.shape[1]
    
    # Expand psi to includ omode dimension
    psi = probe_cplx[:, None, :, :] # (pmode, Ny, Nx) -> (pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices-1):
        object_slice = object_cplx[:, n, :, :] # object_slice -> (omode, Ny, Nx)
        psi = psi * object_slice[None, :, :, :]  # psi -> (pmode, omode, Ny, Nx)
        psi = ifft2(H * fft2(psi, dim=(-2, -1)), dim=(-2, -1))
    
    #Interacting with the last layer, and no propagation is needed afterward
    object_slice = object_cplx[:, n_slices-1, :, :] # object_slice -> (omode, Ny, Nx)
    psi = psi * object_slice[None, :, :, :]  # psi -> (pmode, omode, Ny, Nx)

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from incoherent probe / object modes are summed together
    dp_fwd = torch.sum(torch.square(torch.abs(fftshift(fft2(psi, dim=(-2, -1)), dim=(-2, -1)))), dim=(0, 1))
    return dp_fwd