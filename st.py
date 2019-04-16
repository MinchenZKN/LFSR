import numpy as np
import scipy.ndimage as spy


# eigendecomposition of the structure tensor.
def eigendecomposition(Dx, Dy):
    a11 = np.sum(Dx ** 2, -1)
    a12 = np.sum(Dx * Dy, -1)
    a22 = np.sum(Dy ** 2, -1)

    # disparity
    d = np.tan(0.5 * np.arctan2((2 * a12), (a11 - a22)))
    # coherence
    c = ((a11 - a22) ** 2 + 4 * a12 ** 2) / (a11 + a22) ** 2
    return d, c


def st1_sx(I, tau=0, sigma=0):
    assert I.ndim == 4
    # smooth each channel separately
    I = spy.filters.gaussian_filter(I, [tau, tau, tau, 0])
    # DY
    D = np.zeros((3, 1, 3, 1))
    '''
    # central difference
    D[0, ...] = 1
    D[1, ...] = 0
    D[2, ...] = -1
    '''
    # Scharr diff.
    D[0, :, 0, :] = 3
    D[0, :, 1, :] = 10
    D[0, :, 2, :] = 3
    D[2, :, 0, :] = -3
    D[2, :, 1, :] = -10
    D[2, :, 2, :] = -3
    Dy = spy.convolve(I, D)
    Dy = spy.filters.gaussian_filter(Dy, [sigma, sigma, sigma, 0])

    # DX
    D[:] = 0
    '''
    # central difference
    D[..., 0, :] = 1
    D[..., 1, :] = 0
    D[..., 2, :] = -1
    '''
    # Scharr diff.
    D[0, :, 0, :] = 3
    D[1, :, 0, :] = 10
    D[2, :, 0, :] = 3
    D[0, :, 2, :] = -3
    D[1, :, 2, :] = -10
    D[2, :, 2, :] = -3
    Dx = spy.convolve(I, D)
    Dx = spy.filters.gaussian_filter(Dx, [sigma, sigma, sigma, 0])

    return eigendecomposition(Dx, Dy)


def st1_ty(I,tau = 0, sigma = 0):
    assert I.ndim == 4
    # smooth each channel separately
    I = spy.filters.gaussian_filter(I, [tau, tau, tau, 0])
    # DY
    D = np.zeros((3, 3, 1, 1))
    '''
    # central difference
    D[0, ...] = 1
    D[1, ...] = 0
    D[2, ...] = -1
    '''
    # Scharr diff.
    D[0, 0, ...] = 3
    D[0, 1, ...] = 10
    D[0, 2, ...] = 3
    D[2, 0, ...] = -3
    D[2, 1, ...] = -10
    D[2, 2, ...] = -3
    Dy = spy.convolve(I, D)
    Dy = spy.filters.gaussian_filter(Dy, [sigma, sigma, sigma, 0])


    # DX
    D[:] = 0
    '''
    # central difference
    D[:, 0, ...] = 1
    D[:, 1, ...] = 0
    D[:, 2, ...] = -1
    '''
    # Scharr diff.
    D[0, 0, ...] = 3
    D[1, 0, ...] = 10
    D[2, 0, ...] = 3
    D[0, 2, ...] = -3
    D[1, 2, ...] = -10
    D[2, 2, ...] = -3
    Dx = spy.convolve(I, D)
    Dx = spy.filters.gaussian_filter(Dx, [sigma, sigma, sigma, 0])

    return eigendecomposition(Dx, Dy)
