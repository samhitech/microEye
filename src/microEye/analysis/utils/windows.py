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


def hamming_rect(width: int, height: int):
    '''
    Generate a 2D Hamming window with rectangular shape.

    Parameters
    ----------
    width : int
        Width of the window
    height : int
        Height of the window

    Returns
    -------
    np.ndarray
        2D Hamming window
    '''
    window1d_x = np.hamming(width)
    window1d_y = np.hamming(height)
    return np.sqrt(np.outer(window1d_y, window1d_x))
