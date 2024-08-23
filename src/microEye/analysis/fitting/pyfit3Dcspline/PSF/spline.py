import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

from microEye.analysis.fitting.pyfit3Dcspline.PSF.rubost_mean import robust_mean


class Localization:
    def __init__(self) -> None:
        self.frames = None
        self.frame_z0 = None
        self.rois = None

        self.fit_x = None
        self.fit_y = None
        self.fit_sigmax = None
        self.fit_sigmay = None
        self.fit_intensity = None
        self.fit_background = None


def get_spline_fits(beads: list[Localization], **kwargs):
    # Initialize an empty list to store information about each curve
    curves = []
    dz = kwargs.get('dz', 10)

    # Iterate over beads in reverse order
    for bead in beads:
        beadz0 = bead.frame_z0 * dz

        if 'astig' in kwargs.get('zcorr', '') or 'corr' in kwargs.get('zcorr', ''):
            beadz = (bead.frames * dz) - beadz0
        else:
            beadz = (bead.frames - kwargs['midpoint']) * dz

        sx = bead.fit_sigmax
        sy = bead.fit_sigmay
        z = beadz
        phot = bead.fit_intensity

        # Filter values based on the specified z range
        mask = (z >= kwargs['gaussrange'][0]) & (z <= kwargs['gaussrange'][1])

        # Store information about the current curve
        curves.append({
            'sx': np.array(sx[mask]),
            'sy': np.array(sy[mask]),
            'z': np.array(z[mask]),
            'phot': np.array(phot[mask]),
            'xpos': np.mean(bead.fit_x),
            'ypos': np.mean(bead.fit_y),
        })

    # Get calibrations
    kwargs['ax'] = kwargs['ax_z']
    spline, indgood = clean_and_get_spline(curves, **kwargs)

    return spline, indgood, curves


def clean_and_get_spline(curves, **kwargs):
    """
    Cleans the curves, fits splines, and returns the results.

    Parameters:
    - curves: List of curves containing 'sx', 'sy', and 'z'.
    - p: Dictionary of parameters.

    Returns:
    - spline_result: Dictionary with 'x', 'y', 'zrange', and 'maxmaxrange'.
    - indgood_result: Boolean array indicating 'good' curves.
    """
    za = np.concatenate([curve['z'] for curve in curves])
    Sxa = np.concatenate([curve['sx'] for curve in curves])
    Sya = np.concatenate([curve['sy'] for curve in curves])

    indz = (za > kwargs['gaussrange'][0]) & (za < kwargs['gaussrange'][2])
    z = za[indz]
    Sx = Sxa[indz]
    Sy = Sya[indz]

    splinex = get_spline_interpolator(Sx, z, np.ones_like(z))
    spliney = get_spline_interpolator(Sy, z, np.ones_like(z))

    indgood2 = np.ones(len(curves), dtype=bool)

    zg = np.concatenate([curve['z'][indgood2[i]] for i, curve in enumerate(curves)])
    indz = (zg > kwargs['gaussrange'][0]) & (zg < kwargs['gaussrange'][2])
    zg = zg[indz]
    sxg = np.concatenate([curve['sx'][indgood2[i]] for i, curve in enumerate(curves)])
    syg = np.concatenate([curve['sy'][indgood2[i]] for i, curve in enumerate(curves)])
    sxg = sxg[indz]
    syg = syg[indz]

    splinex2 = get_spline_interpolator(sxg, zg, 1. / (np.abs(sxg - splinex(zg)) + 0.1))
    spliney2 = get_spline_interpolator(syg, zg, 1. / (np.abs(syg - spliney(zg)) + 0.1))

    zt = np.arange(min(zg), max(zg) + 0.01, 0.01)

    plot_curves_and_spline(curves, indgood2, splinex2, spliney2, zt)

    return {
        'x': splinex2, 'y': spliney2,
        'zrange': [zt[0], zt[-1]],
        'maxmaxrange': [min(zg), max(zg)]}, indgood2


def get_spline_interpolator(S, z, weights, smoothing_param=0.96):
    '''
    Returns a 1D cubic spline interpolator for given data.

    Parameters:
    - S: Values to be interpolated.
    - z: Corresponding positions.
    - weights: Weights for each data point.
    - smoothing_param: Smoothing parameter for the spline.

    Returns:
    - spline: Cubic spline interpolator.
    '''
    sorted_indices = np.argsort(z)
    sorted_z = np.sort(z)
    sorted_S = S[sorted_indices]
    sorted_weights = weights[sorted_indices]

    spline = UnivariateSpline(
        sorted_z, sorted_S, w=sorted_weights, k=3, s=smoothing_param)

    return spline

def plot_curves_and_spline(curves, indgood2, spline_x, spline_y, z_range):
    """
    Plots the curves, distinguished between 'good' and 'bad', and the splines.

    Parameters:
    - curves: List of curves containing 'sx', 'sy', and 'z'.
    - indgood2: Boolean array indicating whether each curve is 'good'.
    - spline_x: Interpolator for the x-coordinate.
    - spline_y: Interpolator for the y-coordinate.
    - z_range: Range of z values for plotting.
    """
    z1a, z2a, x1a, x2a = [], [], [], []

    for k, curve in enumerate(curves):
        if indgood2[k]:
            z1a.extend(curve['z'])
            x1a.extend(curve['sx'])
        else:
            z2a.extend(curve['z'])
            x2a.extend(curve['sx'])

    if not z2a:  # If z2a is empty, plot a bad point to have the legend correct
        z2a = [0]
        x2a = [0]

    plt.plot(z2a, x2a, 'g.', label='Bad Curves')
    plt.plot(z1a, x1a, 'r.', label='Good Curves')
    plt.plot(z_range, spline_x(z_range), 'k', label='Spline X')
    plt.plot(z_range, spline_y(z_range), 'k', label='Spline Y')

    plt.xlim([z_range[0], z_range[-1]])
    plt.ylim([0, min(5, max(max(spline_x(z_range)), max(spline_y(z_range))))])
    plt.xlabel('z (nm)')
    plt.ylabel('PSFx, PSFy (pixel)')
    plt.title('Lateral size of the PSF')
    plt.legend()
    plt.show()
