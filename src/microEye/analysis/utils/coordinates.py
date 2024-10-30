
import numpy as np


def radial_coordinate(shape):
    '''Generates a 2D array with radial coordinates
    with according to the first two axis of the
    supplied shape tuple

    Returns
    -------
    R, Rsq : Radius 2d matrix (R) and radius squared matrix (Rsq)
    '''
    y_len = np.arange(-shape[0]//2, shape[0]//2)
    x_len = np.arange(-shape[1]//2, shape[1]//2)
    X, Y = np.meshgrid(x_len, y_len)
    Rsq = (X**2 + Y**2)
    return np.sqrt(Rsq), Rsq
