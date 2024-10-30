import numpy as np


def hamming_2Dwindow(size: int):
    '''
    Generate a 2D Hamming window.

    Parameters
    ----------
    size : int
        Size of the window

    Returns
    -------
    np.ndarray
        2D Hamming window
    '''
    window1d = np.hamming(size)
    return np.sqrt(np.outer(window1d, window1d))
