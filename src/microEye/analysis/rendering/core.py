import numba
import numpy as np


def model(xc, yc, sigma_x, sigma_y, flux, offset, X, Y):
    '''
    2D Gaussian model function.

    Parameters
    ----------
    xc : float
        x-coordinate of the center
    yc : float
        y-coordinate of the center
    sigma_x : float
        Standard deviation in x-direction
    sigma_y : float
        Standard deviation in y-direction
    flux : float
        Total flux
    offset : float
        Offset
    X : np.ndarray
        X-coordinate grid
    Y : np.ndarray
        Y-coordinate grid
    '''
    y_gauss = gauss_1d(Y[:, 0], yc, sigma_y)
    x_gauss = gauss_1d(X[0, :], xc, sigma_x)
    return flux * np.einsum('i,j->ij', y_gauss, x_gauss) + offset

@numba.njit(cache=True)
def gauss_1d(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    '''
    1D Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        x-coordinate
    mu : float
        Mean
    sigma : float
        Standard deviation
    '''
    return 1 / (np.sqrt(2 * np.pi) * sigma) * \
           np.exp(-0.5 * (x - mu)**2 / sigma**2)
