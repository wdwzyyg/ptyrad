import torch

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

## The pseudo-complex functions are adapted from SciComPty's `cplx_lib.py`
## https://www.mdpi.com/2410-3896/6/4/36

def cplx_from_2_arr(re, im, ndim = -1):
    '''Stack two tensor on a new given dimension'''
    # re: real-valued torch Tensor representing the real part of the final pseudo-complex torch Tensor
    # im: real-valued torch Tensor representing the imag part of the final pseudo-complex torch Tensor
    # ndim: The axis that stacks the real and imag part
    # return: pseuso-complex array shaped (...,2)
    
    re, im = re.unsqueeze(ndim), im.unsqueeze(ndim)
    return torch.cat((re, im), ndim)

def cplx_from_np(a, cplx_type, ndim = -1):
    """Transform a complex numpy array in a "complex" tensor"""
    # a: Input complex np array
    # ndim: The axis that stacks the real/imag or amp/phase part
    # cplx_type: "real_imag" or "amp_phase"
    # return: pseuso-complex array shaped (...,2)
    
    if cplx_type == "real_imag":
        return cplx_from_2_arr(torch.from_numpy(a).real, torch.from_numpy(a).imag, ndim)
    elif cplx_type == "amp_phase":
        return cplx_from_2_arr(torch.from_numpy(a).abs(), torch.from_numpy(a).angle(), ndim)

def cplx_mul_2d(h, k):
    """Complex element-wise multiplication between two tensor"""
    # h, k: input pseudo-complex torch Tensor
    # Assuming the last dimension corresponds to the real (0) and imag (1) part
    a, b = h[...,0], h[...,1]
    c, d = k[...,0], k[...,1]
    return cplx_from_2_arr(a*c-b*d, a*d+b*c)