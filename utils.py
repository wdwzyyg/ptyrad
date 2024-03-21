import torch
import warnings
from torch.fft import fft2, ifft2, ifftshift, fftshift


def complex_object_interp3d(complex_object, zoom_factors, z_axis, use_np_or_cp='np'):
    """
    Interpolate a 3D complex object while preserving multiscattering behavior.

    Parameters:
    - complex_object (ndarray): Input complex object with shape (z, y, x).
    - zoom_factors (tuple): Tuple of zoom factors for (z, y, x).
    = z_axis: int indicating the z-axis posiiton
    - use_np_or_cp (str): Specify the library to use, 'np' for NumPy or 'cp' for CuPy.

    Returns:
    ndarray: Interpolated complex object with the same dtype as the input.

    Notes:
    - Amplitude and phase are treated separately as they obey different conservation laws.
    - Phase shift for multiple z-slices is additive, ensuring the sum of all z-slices remains the same.
    - Amplitude between each z-slice is multiplicative. Linear interpolation of log(amplitude) is performed
      while maintaining the conservation law.
    - The phase of the object should be unwrapped and smooth.
    - If possible, use cupy for 40x faster speed (I got 1 sec vs 40 sec for 320*320*420 target size in a one-shot calculation on my Quadro P5000)

    Example:
    >>> complex_object = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)
    >>> zoom_factors = (2, 2, 1.5)
    >>> result = complex_object_interp3d(complex_object, zoom_factors, use_np_or_cp='np')
    """
    
    if use_np_or_cp == 'cp':
        import cupy as xp
        from cupyx.scipy import ndimage
        complex_object = xp.array(complex_object)
    else:
        import numpy as xp
        from scipy import ndimage
    
    if zoom_factors == (1,1,1):
        print(f"No interpolation is needed, returning original object with shape = {complex_object.shape}.")
        return complex_object

    else:
        obj_dtype = complex_object.dtype
        obj_a = xp.abs(complex_object)
        obj_p = xp.angle(complex_object)
        
        obj_a_interp = xp.exp(ndimage.zoom(xp.log(obj_a), zoom_factors) / zoom_factors[z_axis])
        obj_p_interp = ndimage.zoom(obj_p, zoom_factors) / zoom_factors[z_axis]
        
        complex_object_interp3d = obj_a_interp * xp.exp(obj_p_interp*1j)
        print(f"The object shape is interpolated to {complex_object_interp3d.shape}.")
        return complex_object_interp3d.astype(obj_dtype)

def imshift(img, shifts, device):
    """
    Generates a shifted image with support for subpixel shifts.
    
    This function shifts a complex-valued input image by applying phase shifts in the Fourier domain,
    accommodating subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. It should be a complex-valued tensor with shape=(..., Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a single tensor as (shift_y, shift_x).
        device (torch.device): The device on which the computation will be performed.

    Outputs:
        shifted_img (torch.Tensor): The shifted image. It has the same shape as the input image, i.e., shape=(..., Ny, Nx).

    Note:
        - The shifts are specified as fractions of a pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
    """

    Ny, Nx = img.shape[-2], img.shape[-1]
    shift_y, shift_x = shifts[0], shifts[1]
    ry, rx = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing='ij')
    ry, rx = ry.to(device), rx.to(device)
    w = -torch.exp(-(2j * torch.pi) * (shift_x * rx / Nx  + shift_y * ry / Ny))
    shifted_img = ifft2(ifftshift(fftshift(fft2(img)) * w))

    return shifted_img

def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images with support for a batch of subpixel shifts.
    
    This function shifts a complex-valued input image by applying phase shifts in the Fourier domain,
    accommodating subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. It should be a complex-valued tensor with shape=(C, Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively.

    Outputs:
        shifted_img (torch.Tensor): The batch of shifted images. It has the same shape as the input batch of images, i.e., shape=(Nb, C, Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are specified as fractions of a pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - For more general usage of multidimensional array with lots of leading dimensions, we could modify the number of singletons in shifts and grid.
    """
    
    # img = (C,Ny,Nx), currently expecting mixed-state complex probe as a (pmode, Ny, Nx) complex64 tensor.
    # shifts = (Nb, 2)
    # grid = (2, Ny, Nx)
    
    shift_y, shift_x = shifts[:, 0, None, None, None], shifts[:, 1, None, None, None] # shift_y = (Nb, 1, 1, 1)
    ky, kx = grid[None, None, 0], grid[None,None, 1]                                  # ky = (1, 1, Ny, Nx)
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx) + shift_y * ky) 
    shifted_img = ifft2(ifftshift(fftshift(fft2(img[None,...])) * w))
    
    return shifted_img

def near_field_evolution(u_0_shape, z, lambd, extent, use_ASM_only=False, use_np_or_cp='np'):
#  FUNCTION  [u_1, H, h, dH] = near_field_evolution(u_0, z, lambda, extent, use_ASM_only)
#  Description: nearfield evolution function, it automatically switch
#  between ASM and Fraunhofer propagation 
#  Translated from Yi's fold_slice Matlab implementation into CuPy and NumPy by Chia-Hao Lee
    
    if use_np_or_cp == 'cp':
        import cupy as xp
    else:
        import numpy as xp

    u_0 = xp.ones(u_0_shape)
    
    H = None
    h = None
    u_1 = None
    dH = None

    if z == 0:
        H = 1
        u_1 = u_0
        return u_1, H, h, dH

    if z == float('inf'):
        return u_1, H, h, dH

    Npix = u_0.shape

    xgrid = xp.linspace(0.5 + (-Npix[0] / 2), 0.5 + (Npix[0] / 2 - 1), Npix[0]) / Npix[0]

    ygrid = xp.linspace(0.5 + (-Npix[1] / 2), 0.5 + (Npix[1] / 2 - 1), Npix[1]) / Npix[1]


    k = 2 * xp.pi / lambd

    extent = xp.array(extent)
    lambd = xp.array(lambd)
    z = xp.array(z)
    Npix = xp.array(Npix)

    F = xp.mean(extent ** 2 / (lambd * z * Npix))
    
    if abs(F) < 1 and not use_ASM_only:
        # Farfield propagation
        print('Farfield regime, F/Npix={:.2f}'.format(float(F)))
        Xrange = xgrid * extent[0]
        Yrange = ygrid * extent[1]
        X, Y = xp.meshgrid(Xrange, Yrange)
        h = xp.exp(1j * k * z + 1j * k / (2 * z) * (X.T ** 2 + Y.T ** 2))
        H = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(h)))
        H = H / xp.abs(H[Npix[0] // 2, Npix[1] // 2])  # Renormalize to conserve flux in the image
    else:
        # Standard ASM
        kx = 2 * xp.pi * xgrid / extent[0] * Npix[0]
        ky = 2 * xp.pi * ygrid / extent[1] * Npix[1]
        Kx, Ky = xp.meshgrid(kx, ky)
        dH = -1j * (Kx.T ** 2 + Ky.T ** 2) / (2 * k)
        H = xp.exp(1j * z * xp.sqrt(k ** 2 - Kx.T ** 2 - Ky.T ** 2))
        h = None

    # Do the ifftshift inside the function so the output has zero frequency at the center
    H = xp.fft.ifftshift(H)
    return u_1, H, h, dH

def cplx_from_np(a, cplx_type='amp_phase', ndim = -1):
    """ Transform a complex numpy array in a "pseudo-complex" tensor"""
    # a: Input complex np array
    # ndim: The axis that stacks the real/imag or amp/phase part
    # cplx_type: "real_imag" or "amp_phase"
    # return: pseuso-complex array shaped (...,2)
    
    if cplx_type == "real_imag":
        return torch.stack([torch.from_numpy(a).real, torch.from_numpy(a).imag], ndim)
    elif cplx_type == "amp_phase":
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)
    else:
        warnings.warn("cplx_type {} not implemented. Defaulting to 'amp_phase'.".format(cplx_type))
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)


###################################### ARCHIVE ##################################################


def prepare_stack_transform(pos, imgshape):
    """ Generating a stack of 3D affine transformaitons """
    # Useful info: https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html

    # Each affine transform corresponds to an input probe position
    # M is a stack that contains all (N, 3, 4) transformations, while N corresponds to omode
    # M_stack.shape (len(pos), N, 3, 4)

    M_stack = torch.zeros(len(pos), imgshape[0], 3, 4)
    
    for ii, (cy, cx) in enumerate(pos): # Note that there's no cz because we don't need translation in z 
        # The affine transformation Mn is constructed as x,y,z or (W, H, D)
        Mn = torch.tensor([[1., 0, 0, cx*2/imgshape[-1]], [0, 1., 0, cy*2/imgshape[-2]], [0, 0, 1., 0], [0, 0, 0, 1.]]) # Note that normalization is needed
        Mn = Mn[:3] # Only the first 3 rows of the 4x4 affine transformation is needed
        M_stack[ii] = Mn.unsqueeze(0)
    return M_stack