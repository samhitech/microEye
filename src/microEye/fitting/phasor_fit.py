
import numpy as np

from ..Rendering import radial_cordinate


def phasor_fit(image: np.ndarray, points: np.ndarray,
               intensity=True, roi_size=7):
    '''Sub-pixel Phasor 2D fit

    More details:
        see doi.org/10.1063/1.5005899 (Martens et al., 2017)
    '''
    if len(points) < 1:
        return None

    sub_fit = np.zeros((points.shape[0], 5), points.dtype)

    if intensity:
        bg_mask, sig_mask = roi_mask(roi_size)

    for r in range(points.shape[0]):
        x, y = points[r, :]
        idx = int(x - roi_size//2)
        idy = int(y - roi_size//2)
        if idx < 0:
            idx = 0
        if idy < 0:
            idy = 0
        if idx + roi_size > image.shape[1]:
            idx = image.shape[1] - roi_size
        if idy + roi_size > image.shape[0]:
            idy = image.shape[0] - roi_size
        roi = image[idy:idy+roi_size, idx:idx+roi_size]
        fft_roi = np.fft.fft2(roi)
        theta_x = np.angle(fft_roi[0, 1])
        theta_y = np.angle(fft_roi[1, 0])
        if theta_x > 0:
            theta_x = theta_x - 2 * np.pi
        if theta_y > 0:
            theta_y = theta_y - 2 * np.pi
        x = idx + np.abs(theta_x) / (2 * np.pi / roi_size)
        y = idy + np.abs(theta_y) / (2 * np.pi / roi_size)
        sub_fit[r, :2] = [x, y]

        magnitudeX = np.abs(fft_roi[0, 1])
        magnitudeY = np.abs(fft_roi[1, 0])

        sub_fit[r, 4] = magnitudeX / magnitudeY

        if intensity:
            idx = int(x - roi_size//2)
            idy = int(y - roi_size//2)
            if idx < 0:
                idx = 0
            if idy < 0:
                idy = 0
            if idx + roi_size > image.shape[1]:
                idx = image.shape[1] - roi_size
            if idy + roi_size > image.shape[0]:
                idy = image.shape[0] - roi_size
            int_roi = image[idy:idy + roi_size, idx:idx + roi_size]

            sub_fit[r, 2:4] = intensity_estimate(int_roi, bg_mask, sig_mask)

    return sub_fit


def roi_mask(roi_size=7):

    roi_shape = [roi_size] * 2
    roi_radius = roi_size / 2

    radius_map, _ = radial_cordinate(roi_shape)

    bg_mask = radius_map > (roi_radius - 0.5)
    sig_mask = radius_map <= roi_radius

    return bg_mask, sig_mask


def intensity_estimate(roi: np.ndarray, bg_mask, sig_mask, percentile=56):

    background_map = roi[bg_mask]
    background = np.percentile(
        background_map, percentile)

    intensity = np.sum(roi[sig_mask]) - (np.sum(sig_mask) * background)

    return background, max(0, intensity)
