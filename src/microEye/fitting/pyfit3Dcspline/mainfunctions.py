import time
import traceback
import numpy as np
import numba as nb
import math
from scipy import ndimage
from numba import cuda

from . import CPU

from .constants import *


if cuda.is_available():

    from . import GPU

    def GPUmleFit_LM(
            data: np.ndarray, fittype: int, PSFparam: np.ndarray,
            varim: np.ndarray, initZ: float, iterations: int = 30):
        '''
        GPU Fit

        Parameters
        ----------
        data : np.ndarray
            image stack in shape (roi_index, roi_size)
        fittype : int
            fitting mode:

                1. fix PSF.
                2. free PSF.
                3. Gauss fit z. (SMAP developers suppressed this)
                4. fit PSFx, PSFy elliptical.
                5. cspline.

        PSFparam : np.ndarray
            paramters for fitters:

                1. fix PSF: PSFxy sigma.
                2. free PSF: start PSFxy.
                3. Gauss fit z parameters:
                    PSFx, Ax, Ay, Bx, By, gamma, d, PSFy.

                4. fit PSFx, PSFy elliptical: start PSFx, PSFy.
                5. cspline: cspline coefficients.
        varim : np.ndarray
            Variance map for CMOS, if None then EMCCD, default is None.
        initZ : float
            z start parameter.
            unit: distance of stack calibration, center-based.
        iterations : int
            number of fit iterations (if None default=30)

        Returns
        -------
        Parameters : np.ndarray
            fitting results in shape (roi_index, column) for each fit mode:

                1. x, y, bg, I, iteration.
                2. x, y, bg, I, sigma, iteration.
                3. ???
                4. x, y, bg, I, sigmax, sigmay, iteration.
                5. x, y, bg, I, z, iteration.
        CRLBs : np.ndarray
            Cramer-Rao Lower Bounds for fitting results
            in shape (roi_index, column).
        LogLikelihood : np.ndarray
            Log Likelihood in shape (roi_index).
        '''
        blockx = 0
        threadx = 0
        PSFSigma = 1
        startParameters = 0
        d_data = None
        Ndim = Nfitraw = sz = cameratype = 0  # default cameratype 0 EMCCD
        noParameters = 0

        # CMOS
        d_varim = None

        # Spline
        spline_xsize = spline_ysize = spline_zsize = 0
        coeff = d_coeff = 0
        # d_Parameters = d_CRLBs = d_LogLikelihood = 0
        BlockSize = BSZ  # for no shared scenario, we don't care!

        # fun with timing
        freq = 0
        togpu = 0
        fit = 0
        fromgpu = 0
        cleanup = 0
        start = stop = 0
        start = time.perf_counter_ns()
        # QueryPerformanceFrequency(&freq)
        # QueryPerformanceCounter(&start)

        # input checks
        sz = math.sqrt(data.shape[1])
        Nfitraw = data.shape[0]

        # if fittype == 1:  # fixed sigma
        #     pass
        # elif fittype == 2:  # free sigma
        #     pass
        if fittype == 4:  # fixed sigmax and sigmay
            PSFSigma = float(PSFparam[0])
        elif fittype == 5:  # spline fit
            coeff = PSFparam.flatten()
            spline_xsize = PSFparam.shape[1]
            spline_ysize = PSFparam.shape[2]
            spline_zsize = PSFparam.shape[3]

        if varim is None:
            cameratype = 0  # EMCCD
        else:
            cameratype = 1  # sCMOS

        # query GPUs
        assert cuda.is_available(), 'No GPU is available.'
        deviceCount = len(cuda.gpus)

        # check if we are allocating more memory than the card has
        availableMemory = cuda.current_context().get_memory_info()[0]
        requiredMemory = 0
        float_size = 4  # in bytes

        if fittype == 1:  # fixed sigma
            requiredMemory += (2*NV_P + 2) * Nfitraw * float_size
        elif fittype == 2:  # free sigma
            requiredMemory += (2*NV_PS + 2) * Nfitraw * float_size
        elif fittype == 4:  # fixed sigmax and sigmay
            requiredMemory += (2*NV_PS2 + 2) * Nfitraw * float_size
        elif fittype == 5:  # spline fit
            requiredMemory += (2*NV_PS + 2) * Nfitraw * float_size
            requiredMemory += \
                spline_xsize * spline_ysize * spline_zsize * 64 * float_size

        if cameratype == 0:  # EMCCD
            requiredMemory += sz * sz * Nfitraw * float_size
        elif cameratype == 1:  # sCMOS
            requiredMemory += 2 * sz * sz * Nfitraw * float_size

        assert requiredMemory < 0.9*availableMemory, \
            'Trying to allocation {:.3f}MB. GPU only has {:.0f}MB.\n'.format(
                requiredMemory/(1024*1024), availableMemory/(1024*1024)
            ) + \
            'Please break your fitting into multiple smaller runs.\n'
        print(
            'Allocating {:.3e}% out of available GPU memory {:.0f}MB.'.format(
                100*requiredMemory/availableMemory,
                availableMemory/(1024*1024)))

        # copy data to device
        d_data = cuda.to_device(data.flatten().astype(np.float32))

        # CMOS
        if cameratype == 1:
            d_varim = cuda.to_device(varim.flatten().astype(np.float32))

        if fittype == 5:
            d_coeff = cuda.to_device(coeff.flatten().astype(np.float32))

        # create output for parameters and CRLBs
        if fittype == 1:  # (x,y,bg,I)
            d_Parameters = cuda.to_device(
                np.zeros((NV_P + 1) * Nfitraw, np.float32))
            d_CRLBs = cuda.to_device(
                np.zeros(NV_P * Nfitraw, np.float32))
        elif fittype == 2:  # (x,y,bg,I,Sigma)
            d_Parameters = cuda.to_device(
                np.zeros((NV_PS + 1) * Nfitraw, np.float32))
            d_CRLBs = cuda.to_device(
                np.zeros(NV_PS * Nfitraw, np.float32))
        elif fittype == 4:  # (x,y,bg,I,Sx,Sy)
            d_Parameters = cuda.to_device(
                np.zeros((NV_PS2 + 1) * Nfitraw, np.float32))
            d_CRLBs = cuda.to_device(
                np.zeros(NV_PS2 * Nfitraw, np.float32))
        elif fittype == 5:  # (x,y,bg,I,z)
            d_Parameters = cuda.to_device(
                np.zeros((NV_PS + 1) * Nfitraw, np.float32))
            d_CRLBs = cuda.to_device(
                np.zeros(NV_PS * Nfitraw, np.float32))

        d_LogLikelihood = cuda.to_device(
                np.zeros(Nfitraw, np.float32))

        # print(
        #     d_data.nbytes +
        #     (d_varim.nbytes if d_varim is not None else 0) +
        #     (d_coeff.nbytes if fittype == 5 else 0) +
        #     d_Parameters.nbytes +
        #     d_CRLBs.nbytes +
        #     d_LogLikelihood.nbytes,
        #     requiredMemory
        # )

        stop = time.perf_counter_ns()
        togpu = stop - start
        print('Data copied to GPU in {:.3f}ms.\n'.format(
            togpu/1e6))
        start = time.perf_counter_ns()

        # setup kernel
        blockx = math.ceil(Nfitraw / BlockSize)
        threadx = BlockSize

        if fittype == 1:  # fit x,y,bg,I
            if cameratype == 0:
                GPU.kernel_MLEFit_LM_EMCCD[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, Nfitraw)
            elif cameratype == 1:
                GPU.kernel_MLEFit_LM_sCMOS[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, int(Nfitraw), d_varim)
        elif fittype == 2:  # fit x,y,bg,I,sigma
            if cameratype == 0:
                GPU.kernel_MLEFit_LM_Sigma_EMCCD[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, int(Nfitraw))
            elif cameratype == 1:
                GPU.kernel_MLEFit_LM_Sigma_sCMOS[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, int(Nfitraw), d_varim)
        elif fittype == 4:  # fit x,y,bg,I,sigmax,sigmay
            if cameratype == 0:
                GPU.kernel_MLEFit_LM_sigmaxy_EMCCD[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, int(Nfitraw))
            elif cameratype == 1:
                GPU.kernel_MLEFit_LM_sigmaxy_sCMOS[blockx, threadx](
                    d_data, PSFSigma, int(sz), iterations, d_Parameters,
                    d_CRLBs, d_LogLikelihood, int(Nfitraw), d_varim)
        elif fittype == 5:  # fit x,y,bg,I,z
            if (initZ < 0):
                initZ = spline_zsize / 2.0

            if cameratype == 0:
                GPU.kernel_splineMLEFit_z_EMCCD[blockx, threadx](
                    d_data, d_coeff, spline_xsize, spline_ysize, spline_zsize,
                    int(sz), iterations,
                    d_Parameters, d_CRLBs, d_LogLikelihood,
                    initZ, int(Nfitraw))
            elif cameratype == 1:
                GPU.kernel_splineMLEFit_z_sCMOS[blockx, threadx](
                    d_data, d_coeff, spline_xsize, spline_ysize, spline_zsize,
                    int(sz), iterations,
                    d_Parameters, d_CRLBs, d_LogLikelihood,
                    initZ, int(Nfitraw), d_varim)

        cuda.synchronize()

        stop = time.perf_counter_ns()
        fit = stop - start
        print('Fitted {:d} localizations in {:.3f}ms.\n'.format(
            Nfitraw,
            fit/1e6))
        start = time.perf_counter_ns()

        # copy to matlab output
        if fittype == 1:  # (x,y,bg,I)
            Parameters = d_Parameters.copy_to_host()
            CRLBs = d_CRLBs.copy_to_host()
        elif fittype == 2:  # (x,y,bg,I,Sigma)
            Parameters = d_Parameters.copy_to_host()
            CRLBs = d_CRLBs.copy_to_host()
        elif fittype == 4:  # (x,y,bg,I,Sx,Sy)
            Parameters = d_Parameters.copy_to_host()
            CRLBs = d_CRLBs.copy_to_host()
        elif fittype == 5:  # (x,y,bg,I,z)
            Parameters = d_Parameters.copy_to_host()
            CRLBs = d_CRLBs.copy_to_host()

        LogLikelihood = d_LogLikelihood.copy_to_host()

        stop = time.perf_counter_ns()
        fromgpu = stop - start
        print('Data copied to Host in {:.3f}ms.\n'.format(
            fromgpu/1e6))
        start = time.perf_counter_ns()

        # cleanup
        # cuda.current_context().reset()

        # cuda.close()

        stop = time.perf_counter_ns()

        # reshape output for parameters and CRLBs
        if fittype == 1:  # (x,y,bg,I)
            Parameters = Parameters.reshape((-1, (NV_P + 1)))
            CRLBs = CRLBs.reshape((-1, NV_P))
        elif fittype == 2:  # (x,y,bg,I,Sigma)
            Parameters = Parameters.reshape((-1, (NV_PS + 1)))
            CRLBs = CRLBs.reshape((-1, NV_PS))
        elif fittype == 4:  # (x,y,bg,I,Sx,Sy)
            Parameters = Parameters.reshape((-1, (NV_PS2 + 1)))
            CRLBs = CRLBs.reshape((-1, NV_PS2))
        elif fittype == 5:  # (x,y,bg,I,z)
            Parameters = Parameters.reshape((-1, (NV_PS + 1)))
            CRLBs = CRLBs.reshape((-1, NV_PS))

        return Parameters, CRLBs, LogLikelihood


def CPUmleFit_LM(
        data: np.ndarray, fittype: int, PSFparam: np.ndarray,
        varim: np.ndarray, initZ: float, iterations: int = 30):
    '''
    CPU Fit

    Parameters
    ----------
    data : np.ndarray
        image stack in shape (roi_index, roi_size)
    fittype : int
        fitting mode:

            1. fix PSF.
            2. free PSF.
            3. Gauss fit z. (SMAP developers suppressed this)
            4. fit PSFx, PSFy elliptical.
            5. cspline.

    PSFparam : np.ndarray
        paramters for fitters:

            1. fix PSF: PSFxy sigma.
            2. free PSF: start PSFxy.
            3. Gauss fit z: parameters: PSFx, Ax, Ay, Bx, By, gamma, d, PSFy.
            4. fit PSFx, PSFy elliptical: start PSFx, PSFy.
            5. cspline: cspline coefficients.
    varim : np.ndarray
        Variance map for CMOS, if None then EMCCD, default is None.
    initZ : float
        z start parameter. unit: distance of stack calibration, center based.
    iterations : int
        number of fit iterations (default=30)

    Returns
    -------
    Parameters : np.ndarray
        fitting results in shape (roi_index, column) for each fit mode:

            1. x, y, bg, I, iteration.
            2. x, y, bg, I, sigma, iteration.
            3. ???
            4. x, y, bg, I, sigmax, sigmay, iteration.
            5. x, y, bg, I, z, iteration.
    CRLBs : np.ndarray
        Cramer-Rao Lower Bounds for fitting results
        in shape (roi_index, column).
    LogLikelihood : np.ndarray
        Log Likelihood in shape (roi_index).
    '''
    PSFSigma = 1
    cameratype = 0  # default cameratype 0 EMCCD
    noParameters = 0

    # Spline
    spline_xsize = spline_ysize = spline_zsize = 0
    coeff = 0

    # fun with timing
    freq = 0
    togpu = 0
    fit = 0
    fromgpu = 0
    cleanup = 0
    start = stop = 0
    start = time.perf_counter_ns()
    # QueryPerformanceFrequency(&freq)
    # QueryPerformanceCounter(&start)

    # input checks
    sz = math.sqrt(data.shape[1])
    Nfitraw = data.shape[0]

    if fittype == 1:  # fixed sigma
        PSFSigma = float(PSFparam[0])
    elif fittype == 2:  # free sigma
        PSFSigma = float(PSFparam[0])
    elif fittype == 4:  # fixed sigmax and sigmay
        PSFSigma = float(PSFparam[0])
    elif fittype == 5:  # spline fit
        coeff = PSFparam.flatten()
        spline_xsize = PSFparam.shape[1]
        spline_ysize = PSFparam.shape[2]
        spline_zsize = PSFparam.shape[3]

    if varim is None:
        cameratype = 0  # EMCCD
    else:
        cameratype = 1  # sCMOS

    # create output for parameters and CRLBs
    if fittype == 1:  # (x,y,bg,I)
        Parameters = np.zeros((Nfitraw, (NV_P + 1)), np.float32)
        CRLBs = np.zeros((Nfitraw, NV_P), np.float32)
    elif fittype == 2:  # (x,y,bg,I,Sigma)
        Parameters = np.zeros((Nfitraw, (NV_PS + 1)), np.float32)
        CRLBs = np.zeros((Nfitraw, NV_PS), np.float32)
    elif fittype == 4:  # (x,y,bg,I,Sx,Sy)
        Parameters = np.zeros((Nfitraw, (NV_PS2 + 1)), np.float32)
        CRLBs = np.zeros((Nfitraw, NV_PS2), np.float32)
    elif fittype == 5:  # (x,y,bg,I,z)
        Parameters = np.zeros((Nfitraw, (NV_PS + 1)), np.float32)
        CRLBs = np.zeros((Nfitraw, NV_PS), np.float32)

    LogLikelihood = np.zeros(Nfitraw, np.float32)

    stop = time.perf_counter_ns()
    togpu = stop - start
    start = time.perf_counter_ns()

    # if fittype == 1:  # fit x,y,bg,I
    #     for ii in range(Nfitraw):
    #         Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
    #             CPU.kernel_MLEFit_LM(
    #                 data[ii, :], PSFSigma, int(sz), iterations,
    #                 varim if varim is None else varim[ii, :])
    # elif fittype == 2:  # fit x,y,bg,I,sigma
    #     for ii in range(Nfitraw):
    #         Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
    #             CPU.kernel_MLEFit_LM_Sigma(
    #                 data[ii, :], PSFSigma, int(sz), iterations,
    #                 varim if varim is None else varim[ii, :])
    # elif fittype == 4:  # fit x,y,bg,I,sigmax,sigmay
    #     for ii in range(Nfitraw):
    #         Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
    #             CPU.kernel_MLEFit_LM_sigmaxy(
    #                 data[ii, :], PSFSigma, int(sz), iterations,
    #                 varim if varim is None else varim[ii, :])
    if fittype == 5:  # fit x,y,bg,I,z
        if (initZ < 0):
            initZ = spline_zsize / 2.0

        CPU_parallel_fit_3D_scpline(
            data, varim,
            coeff, spline_xsize, spline_ysize, spline_zsize,
            sz, Nfitraw, initZ, iterations,
            Parameters, CRLBs, LogLikelihood)
    else:
        CPU_parallel_fit_2D(
            data, fittype, PSFSigma, varim,
            sz, Nfitraw, initZ, iterations,
            Parameters, CRLBs, LogLikelihood)

    stop = time.perf_counter_ns()
    fit = stop - start
    start = time.perf_counter_ns()

    stop = time.perf_counter_ns()
    fromgpu = stop - start
    start = time.perf_counter_ns()

    # cleanup
    stop = time.perf_counter_ns()

    return Parameters, CRLBs, LogLikelihood


def init_numba_CPU():
    n = 21
    sigma = 4
    x = np.arange(n, dtype=np.float32)
    y = 100*np.exp(-(x-n/2)**2/(2*sigma**2))
    y = np.tile(y, n).reshape((n, n))
    z = (y*y.T).flatten().astype(np.float32)
    z = z[np.newaxis, ...]

    varim = np.ones(z.shape, dtype=np.float32)
    LogLikelihood = np.zeros((1, 1), dtype=np.float32)
    # varim=None
    Parameters = np.zeros((1, (NV_P + 1)), np.float32)
    CRLBs = np.zeros((1, NV_P), np.float32)
    CPU_parallel_fit_2D(
        z, 1, 1.5, None, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS), np.float32)
    CPU_parallel_fit_2D(
        z, 2, 1.5, None, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS2 + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS2), np.float32)
    CPU_parallel_fit_2D(
        z, 4, 1.5, None, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS), np.float32)
    CPU_parallel_fit_3D_scpline(
        z, varim, np.ones((64, 4, 4, 4), dtype=np.float32).flatten(),
        int(4), int(4), int(4), int(n), 1, 0, 30,
        Parameters, CRLBs, LogLikelihood)
    # varim not None
    Parameters = np.zeros((1, (NV_P + 1)), np.float32)
    CRLBs = np.zeros((1, NV_P), np.float32)
    CPU_parallel_fit_2D(
        z, 1, 1.5, varim, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS), np.float32)
    CPU_parallel_fit_2D(
        z, 2, 1.5, varim, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS2 + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS2), np.float32)
    CPU_parallel_fit_2D(
        z, 4, 1.5, varim, n, 1, 0, 1,
        Parameters, CRLBs, LogLikelihood)
    Parameters = np.zeros((1, (NV_PS + 1)), np.float32)
    CRLBs = np.zeros((1, NV_PS), np.float32)
    CPU_parallel_fit_3D_scpline(
        z, varim, np.ones((64, 4, 4, 4), dtype=np.float32).flatten(),
        int(4), int(4), int(4), int(n), 1, 0, 30,
        Parameters, CRLBs, LogLikelihood)


@nb.njit(parallel=True)
def CPU_parallel_fit_2D(
        data, fittype, PSFSigma, varim,
        sz, Nfitraw, initZ, iterations,
        Parameters, CRLBs, LogLikelihood):
    if fittype == 1:  # fit x,y,bg,I
        for ii in nb.prange(Nfitraw):
            try:
                Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
                    CPU.kernel_MLEFit_LM(
                        data[ii, :], PSFSigma, int(sz), iterations,
                        None if varim is None else varim[ii, :])
            except Exception:
                Parameters[ii, :] = -1
                CRLBs[ii, :] = -1
                LogLikelihood[ii] = -1
    elif fittype == 2:  # fit x,y,bg,I,sigma
        for ii in nb.prange(Nfitraw):
            try:
                Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
                    CPU.kernel_MLEFit_LM_Sigma(
                        data[ii, :], PSFSigma, int(sz), iterations,
                        None if varim is None else varim[ii, :])
            except Exception:
                Parameters[ii, :] = -1
                CRLBs[ii, :] = -1
                LogLikelihood[ii] = -1
    elif fittype == 4:  # fit x,y,bg,I,sigmax,sigmay
        for ii in nb.prange(Nfitraw):
            try:
                Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
                    CPU.kernel_MLEFit_LM_sigmaxy(
                        data[ii, :], PSFSigma, int(sz), iterations,
                        None if varim is None else varim[ii, :])
            except Exception:
                Parameters[ii, :] = -1
                CRLBs[ii, :] = -1
                LogLikelihood[ii] = -1


@nb.njit(parallel=True)
def CPU_parallel_fit_3D_scpline(
        data, varim,
        coeff, spline_xsize, spline_ysize, spline_zsize,
        sz, Nfitraw, initZ, iterations,
        Parameters, CRLBs, LogLikelihood):
    for ii in nb.prange(Nfitraw):
        try:
            Parameters[ii, :], CRLBs[ii, :], LogLikelihood[ii] = \
                CPU.kernel_splineMLEFit_z(
                    data[ii, :],
                    coeff, spline_xsize, spline_ysize, spline_zsize,
                    int(sz), iterations, initZ,
                    None if varim is None else varim[ii, :])
        except Exception:
            Parameters[ii, :] = -1
            CRLBs[ii, :] = -1
            LogLikelihood[ii] = -1


@nb.njit
def get_roi_list(image: np.ndarray, points: np.ndarray, roi_size=7):
    '''
    Gets the roi list of specific size around the supplied (x, y) points

    Parameters
    ----------
    image : np.ndarray
        The single channel image
    points : np.ndarray
        The points list of preliminary detection
    roi_size : int, optional
        roi size, by default 7

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        roi_list array of shape (nRoi, roi_size**2),
        coord_list of roi top left corner
    '''
    if len(points) < 1:
        return None

    assert len(image.shape) == 2, 'image should be a 2D ndarray!'

    roi_list = np.zeros((points.shape[0], roi_size**2), np.float32)
    coord_list = np.zeros_like(points)

    for r in nb.prange(points.shape[0]):
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
        coord_list[r, :] = [idx, idy]
        roi_list[r, :] = image[idy:idy+roi_size, idx:idx+roi_size].flatten()

    return roi_list, coord_list


@nb.njit
def get_roi_list_CMOS(
        image: np.ndarray, varim: np.ndarray,
        points: np.ndarray, roi_size=7):
    '''
    Gets the roi list of specific size around the supplied (x, y) points

    Parameters
    ----------
    image : np.ndarray
        The single channel image
    varim : np.ndarray
        The CMOS pixel variance map
    points : np.ndarray
        The points list of preliminary detection
    roi_size : int, optional
        roi size, by default 7

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        roi_list array of shape (nRoi, roi_size**2),
        coord_list of roi top left corner
    '''
    if len(points) < 1:
        return None

    assert len(image.shape) == 2, 'image should be a 2D ndarray!'

    assert image.shape != varim.shape, 'image & varim must have equal shape!'

    roi_list = np.zeros((2, points.shape[0], roi_size**2), np.float32)
    coord_list = np.zeros_like(points)

    for r in nb.prange(points.shape[0]):
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
        coord_list[r, :] = [idx, idy]
        roi_list[0, r, :] = image[idy:idy+roi_size, idx:idx+roi_size].flatten()
        roi_list[1, r, :] = varim[idy:idy+roi_size, idx:idx+roi_size].flatten()

    return roi_list[0], roi_list[1], coord_list


def psf2cspline_np(psf):
    # calculate A
    A = np.zeros((64, 64))
    for i in range(1, 5):
        dx = (i-1)/3
        for j in range(1, 5):
            dy = (j-1)/3
            for k in range(1, 5):
                dz = (k-1)/3
                for ll in range(1, 5):
                    for m in range(1, 5):
                        for n in range(1, 5):
                            A[
                                (i-1)*16+(j-1)*4+k - 1,
                                (ll-1)*16+(m-1)*4+n - 1] = \
                                    dx**(ll-1) * dy**(m-1) * dz**(n-1)

    # upsample psf with factor of 3
    psf_up = ndimage.zoom(
        psf, 3.0,
        mode='grid-constant',
        grid_mode=True)[1:-1, 1:-1, 1:-1]
    A = np.float32(A)
    coeff = calsplinecoeff(A, psf, psf_up)
    return coeff


def calsplinecoeff(A, psf, psf_up):
    # calculate cspline coefficients
    coeff = np.zeros((64, psf.shape[0]-1, psf.shape[1]-1, psf.shape[2]-1))
    for i in range(coeff.shape[1]):
        for j in range(coeff.shape[2]):
            for k in range(coeff.shape[3]):
                temp = psf_up[
                    i*3: 3*(i+1)+1,
                    j*3: 3*(j+1)+1,
                    k*3: 3*(k+1)+1]
                x = np.linalg.solve(A, temp.flatten())
                coeff[:, i, j, k] = x

    return coeff
