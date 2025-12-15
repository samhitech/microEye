import numpy as np

from microEye.qt import QtCore, QtGui, QtWidgets, Signal


def wavelength_to_rgb(WL, GAMMA=0.8, max_intensity=255):
    '''
    Calculates approximate R, G, B values for a given wavelength (WL) in nm,
    using the linear approximation method described in the Dan Bruton Fortran program.
    Returns RGB values as floats in the range [0.0, 1.0], with gamma applied.
    '''
    R, G, B = 0.0, 0.0, 0.0
    SSS = 1.0  # SSS is the intensity factor near spectrum limits

    # Determine base R, G, B components (linear ramps)
    if 380.0 <= WL <= 440.0:
        R = -1.0 * (WL - 440.0) / (440.0 - 380.0)
        B = 1.0
    elif 440.0 <= WL <= 490.0:
        G = (WL - 440.0) / (490.0 - 440.0)
        B = 1.0
    elif 490.0 <= WL <= 510.0:
        G = 1.0
        B = -1.0 * (WL - 510.0) / (510.0 - 490.0)
    elif 510.0 <= WL <= 580.0:
        R = (WL - 510.0) / (580.0 - 510.0)
        G = 1.0
    elif 580.0 <= WL <= 645.0:
        R = 1.0
        G = -1.0 * (WL - 645.0) / (645.0 - 580.0)
    elif 645.0 <= WL <= 780.0:
        R = 1.0

    # Adjust intensity near the limits of visible spectrum (Dan Bruton's SSS factor)
    if WL > 700.0:
        SSS = 0.3 + 0.7 * (780.0 - WL) / (780.0 - 700.0)
    elif WL < 420.0:
        SSS = 0.3 + 0.7 * (WL - 380.0) / (420.0 - 380.0)
    # else SSS remains 1.0

    # Apply intensity and gamma correction
    # Using numpy power function for clean handling of potential edge cases
    final_r = (SSS * R) ** GAMMA
    final_g = (SSS * G) ** GAMMA
    final_b = (SSS * B) ** GAMMA

    # Clamp values to [0.0, 1.0] just in case
    final_r = np.clip(final_r, 0.0, 1.0)
    final_g = np.clip(final_g, 0.0, 1.0)
    final_b = np.clip(final_b, 0.0, 1.0)

    # Scale to max_intensity
    final_r = int(final_r * max_intensity)
    final_g = int(final_g * max_intensity)
    final_b = int(final_b * max_intensity)

    return final_r, final_g, final_b
