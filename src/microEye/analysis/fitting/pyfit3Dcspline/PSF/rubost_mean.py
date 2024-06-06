import numpy as np
from scipy.optimize import fmin


def robust_mean(
        data: np.ndarray, axis: int=None, k_sigma: float=3, fit_mean: bool=False):
    '''
    Calculate robust mean, standard deviation, and indices of inliers and outliers.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
    axis : int, optional
        Axis along which the mean is taken. Default is None.
    k_sigma : float, optional
        Number of sigmas at which to place the cut-off. Default is 3.
    fit_mean : bool, optional
        Whether or not to use fitting to robustly estimate the mean.
        Default is False.
        If True, mean is approximated by minimizing the median deviation.
        If False, mean is approximated by the median.

    Returns
    -------
    final_mean : float
        Robust mean.
    std_sample : float
        Standard deviation of the data (divide by sqrt(n) to get std of the mean).
    inlier_idx : numpy.ndarray
        Index into data with the inliers.
    outlier_idx : numpy.ndarray
        Index into data with the outliers.

    Warning
    -------
    NaN or Inf will be counted as neither in- nor outlier.
    The code is based on (linear) Least Median Squares and
    can be changed to include weights.

    References
    ----------
    J. Ries https://github.com/jries/SMAP

    Example
    -------
    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> data = np.concatenate([norm.rvs(loc=0, scale=1, size=100),
    ...                         norm.rvs(loc=10, scale=1, size=5)])
    >>> final_mean, std_sample, inlier_idx, outlier_idx = robust_mean(data)
    '''
    if data.size == 0:
        raise ValueError('Parameter data should be non-empty ndarray!')

    if axis is None:
        # make sure that the axis is correct if there's a vector
        if np.any(np.array(data.shape) == 1) and len(data.shape) == 2:
            axis = np.where(np.array(data.shape) > 1)[0][0]
        else:
            axis = 0

    if k_sigma is None:
        k_sigma = 3

    if fit_mean and len(data.shape) > 1:
        raise ValueError(f'Fitting {len(data.shape)}-D data is not supported!')

    if np.sum(np.isfinite(data)) < 4:
        print('Warning: Less than 4 finite data points!')
        finite_data = data[np.isfinite(data)]
        final_mean = np.nanmean(finite_data, axis=axis)
        std_sample = np.nanstd(finite_data, axis=axis)
        inlier_idx = np.where(np.isfinite(data))
        outlier_idx = np.array([], dtype=int)
        return final_mean, std_sample, inlier_idx, outlier_idx

    # LEAST MEDIAN SQUARES
    # define magic numbers:
    magic_number2 = 1.4826**2  # see Danuser, 1992 or Rousseeuw & Leroy, 1987

    # store data size and reduced data size
    data_size = np.array(data.shape)
    reduced_data_size = data_size.copy()
    reduced_data_size[axis] = 1

    # need this for later
    blow_up_data_size = data_size // reduced_data_size

    # count relevant dimensions besides choosen axis
    real_dimensions = np.sum(data_size > 1)

    # compute median
    if fit_mean:
        # minimize the median deviation from the mean
        median_data = fmin(lambda x: np.median(np.abs(data - x)), np.median(data))
    else:
        median_data = np.nanmedian(data, axis=axis, keepdims=True)

    # calculate statistics
    res2 = (data - np.tile(median_data, blow_up_data_size))**2
    med_res2 = np.maximum(
        np.nanmedian(res2, axis=axis, keepdims=True), np.finfo(float).eps)

    # test value to calculate weights
    test_value = res2 / np.tile(magic_number2 * med_res2, blow_up_data_size)

    if real_dimensions == 1:
        # goodRows: weight 1, badRows: weight 0
        inlier_idx = np.where(test_value <= k_sigma**2)
        outlier_idx = np.where(test_value > k_sigma**2)

        # calculate std of the sample
        if len(inlier_idx[0]) > 4:
            std_sample = np.sqrt(np.sum(res2[inlier_idx]) / (len(inlier_idx[0]) - 4))
        else:
            std_sample = np.nan

        # MEAN
        final_mean = np.mean(data[inlier_idx])

    else:
        # goodRows: weight 1, badRows: weight 0
        inlier_idx = np.where(test_value <= k_sigma**2)
        outlier_idx = np.where(test_value > k_sigma**2)

        # mask outliers
        res2[outlier_idx] = np.nan

        # count inliers
        n_inliers = np.sum(~np.isnan(res2), axis=axis, keepdims=True)

        # calculate std of the sample

        # put NaN wherever there are not enough data points to calculate a
        # standard deviation
        good_idx = np.sum(np.isfinite(res2), axis=axis, keepdims=True) > 4
        good_idx_broadcasted = np.broadcast_to(good_idx, res2.shape)
        std_sample = np.full_like(good_idx, np.nan, dtype=float).squeeze()
        if np.any(good_idx_broadcasted):
            std_sample[:] = np.sqrt(
                np.nansum(
                    res2 * good_idx_broadcasted, axis=axis) / (n_inliers[good_idx] - 4)
            ).squeeze()

        # MEAN
        data[outlier_idx] = np.nan
        final_mean = np.nanmean(data, axis=axis)

    return final_mean, std_sample, inlier_idx, outlier_idx
