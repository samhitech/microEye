
import math

import numba as nb
from numba import cuda

from microEye.analysis.fitting.pyfit3Dcspline.constants import NV_PSP
from microEye.analysis.fitting.pyfit3Dcspline.GPU.GPUfunctions import *


@cuda.jit(device=True)
def kernel_computeDelta3D(
        x_delta, y_delta, z_delta, delta_f, delta_dxf, delta_dyf, delta_dzf):
    '''
    This function for calculation of the commonterm for
    Cspline is adpopted from:
    "Analyzing Single Molecule Localization Microscopy
    Data Using Cubic Splines", Hazen Babcok, Xiaowei Zhuang,
    Scientific Report, 1, 552 , 2017.

    Parameters
    ----------
    x_delta : float
        _description_
    y_delta : float
        _description_
    z_delta : float
        _description_
    delta_f : float
        _description_
    delta_dxf : float[]
        _description_
    delta_dyf : float[]
        _description_
    delta_dzf : float[]
        _description_

    Returns
    -------
    tuple[...]
        delta_dxf, delta_dyf, delta_dzf
    '''

    # int i,j,k;
    # float cx,cy,cz;

    cz = ONE
    for i in range(4):
        cy = ONE
        for j in range(4):
            cx = ONE
            for k in range(4):
                delta_f[i*16 + j*4 + k] = cz * cy * cx
                if(k < 3):
                    delta_dxf[i*16+j*4+k+1] = nb.float32(k + 1) * cz * cy * cx

                if(j < 3):
                    delta_dyf[i*16+(j+1)*4+k] = nb.float32(j + 1) * cz * cy * cx

                if(i < 3):
                    delta_dzf[(i+1)*16+j*4+k] = nb.float32(i + 1) * cz * cy * cx

                cx = cx * x_delta
            cy = cy * y_delta
        cz = cz * z_delta

    return delta_f, delta_dxf, delta_dyf, delta_dzf


@cuda.jit(device=True)
def kernel_cholesky(A, n, L, U) -> int:
    '''
    Parameters
    ----------
    A : float[]
        _description_
    n : int
        _description_
    L : float[]
        _description_
    U : float[]
        _description_

    Returns
    -------
    int
        _description_
    '''
    info = 0
    for i in range(n):
        for j in range(i+1):
            s = ZERO
            for k in range(j):
                s += U[i * n + k] * U[j * n + k]

            if (i == j):
                if (A[i*n+i]-s >= ZERO):
                    U[i * n + j] = math.sqrt(A[i * n + i] - s)
                    L[j*n+i] = U[i * n + j]
                else:
                    info = 1
                    return info
            else:
                U[i * n + j] = (ONE / U[j * n + j] * (A[i * n + j] - s))
                L[j*n+i] = U[i * n + j]
    return info


@cuda.jit(device=True)
def kernel_luEvaluate(L, U, b, n, x):
    '''_summary_

    Parameters
    ----------
    L : float
        _description_
    U : float
        _description_
    b : float
        _description_
    n : int
        _description_
    x : float
        _description_

    Returns
    -------
    float[...]
        x
    '''
    # Ax = b -> LUx = b. Then y is defined to be Ux
    # for sigmaxy, we have 6 parameters
    y = cuda.local.array(6, nb.float32)

    # Forward solve Ly = b
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[j*n+i] * y[j]
        y[i] /= L[i*n+i]

    # Backward solve Ux = y
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n, 1):
            x[i] -= U[j*n+i] * x[j]
        x[i] /= U[i*n + i]

    return x


@cuda.jit(
    nb.types.Tuple((nb.float32[:], nb.float32))(
        nb.int32, nb.int32, nb.int32,
        nb.int32, nb.int32, nb.int32,
        nb.float32[:], nb.float32[:], nb.float32[:],
        nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:]),
    device=True)
def kernel_DerivativeSpline(
        xc, yc, zc, xsize, ysize, zsize,
        delta_f, delta_dxf, delta_dyf, delta_dzf, coeff, theta, dudt):
    '''
    Parameters
    ----------
    xc : int
        _description_
    yc : int
        _description_
    zc : int
        _description_
    xsize : int
        _description_
    ysize : int
        _description_
    zsize : int
        _description_
    delta_f : float
        _description_
    delta_dxf : float
        _description_
    delta_dyf : float
        _description_
    delta_dzf : float
        _description_
    coeff : float
        _description_
    theta : float
        _description_
    dudt : float
        _description_

    Returns
    -------
    tuple[...]
        dudt, model
    '''
    temp = ZERO
    # the caller reuses dudt for every pixel (the CPU version makes a new one)
    for i in range(NV_PSP):
        dudt[i] = ZERO

    xc = max(xc, 0)
    xc = min(xc, xsize - 1)

    yc = max(yc, 0)
    yc = min(yc, ysize - 1)

    zc = max(zc, 0)
    zc = min(zc, zsize - 1)

    for i in range(64):
        temp += (
            delta_f[i] *
            coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc])
        dudt[0] += (
            delta_dxf[i] *
            coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc])
        dudt[1] += (
            delta_dyf[i] *
            coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc])
        dudt[4] += (
            delta_dzf[i] *
            coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc])

    dudt[0] *= -theta[2]
    dudt[1] *= -theta[2]
    dudt[4] *= theta[2]
    dudt[2] = temp
    dudt[3] = ONE

    model = theta[3] + theta[2] * max(TINY, temp)

    return dudt, model
