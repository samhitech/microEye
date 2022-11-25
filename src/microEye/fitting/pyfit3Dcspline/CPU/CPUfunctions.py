
import numpy as np
import numba as nb
import math

from ..constants import *


@nb.njit(
    nb.float32(nb.int32, nb.float32, nb.float32)
)
def kernel_IntGauss1D(ii, x, sigma):
    '''_summary_

    Parameters
    ----------
    ii : int
        ???
    x : float
        ???
    sigma : float
        sigma value of the PSF
    '''
    norm = 1.0 / (2.0*sigma*sigma)

    return 0.5 * (
        math.erf((ii-x+0.5) * math.sqrt(norm))
        - math.erf((ii-x-0.5) * math.sqrt(norm)))


@nb.njit(
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32)
)
def kernel_alpha(z, Ax, Bx, d):
    '''compute coefficient for alpha

    Parameters
    ----------
    z : float
        ???
    Ax : float
        ???
    Bx : float
        ???
    d : float
        ???

    Returns
    -------
    float
        alpha value
    '''

    return 1.0 + math.pow(z/d, 2) + Ax*math.pow(z/d, 3) + Bx*math.pow(z/d, 4)


@nb.njit(
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32)
)
def kernel_dalphadz(z, Ax, Bx, d) -> float:
    '''compute first derivative for alpha in relation to z

    Parameters
    ----------
    z : float
        ???
    Ax : float
        ???
    Bx : float
        ???
    d : float
        ???

    Returns
    -------
    float
        first derivative for alpha value
    '''

    return (
        2.0 * z/(d*d) +
        3.0*Ax*math.pow(z, 2)/(d*d*d) +
        4.0*Bx*math.pow(z, 3)/math.pow(d, 4))


@nb.njit(
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32)
)
def kernel_d2alphadz2(z, Ax, Bx, d):
    '''compute second derivative for alpha in relation to z

    Parameters
    ----------
    z : float
        ???
    Ax : float
        ???
    Bx : float
        ???
    d : float
        ???

    Returns
    -------
    float
        second derivative for alpha value
    '''

    return (
        2.0/(d*d) +
        6.0*Ax*z/(d*d*d) +
        12.0*Bx*math.pow(z, 2)/math.pow(d, 4))


@nb.njit(
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int32, nb.float32, nb.float32,
        nb.float32, nb.float32, nb.boolean)
)
def kernel_DerivativeIntGauss1D(ii, x, sigma, N, PSFy, calc_d2):
    '''compute the derivative of the 1D gaussian

    Parameters
    ----------
    ii : int
        ???
    x : float
        ???
    sigma : float
        ???
    N : float
        ???
    PSFy : float
        ???
    calc_d2 : bool
        calculate d2udt2

    Returns
    -------
    (float, float)
        dudt, d2udt2 values
    '''
    a = math.exp(-0.5*math.pow(((ii+0.5-x)/sigma), 2))
    b = math.exp(-0.5*math.pow((ii-0.5-x)/sigma, 2))

    dudt = - (N / (math.sqrt(2.0 * pi) * sigma)) * (a-b) * PSFy
    d2udt2 = 0

    if (calc_d2):
        d2udt2 = - (N / (math.sqrt(2.0 * pi) * math.pow(sigma, 3))) * \
                 ((ii+0.5-x)*a-(ii-0.5-x)*b)*PSFy

    return dudt, d2udt2


@nb.njit(
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int32, nb.float32, nb.float32,
        nb.float32, nb.float32, nb.boolean)
)
def kernel_DerivativeIntGauss1DSigma(ii, x, Sx, N, PSFy, calc_d2):
    '''compute the derivative of the 1D gaussian

    Parameters
    ----------
    ii : int
        ???
    x : float
        ???
    sigma : float
        ???
    N : float
        ???
    PSFy : float
        ???
    calc_d2 : bool
        calculate d2udt2

    Returns
    -------
    (float, float)
        dudt, d2udt2 values
    '''
    ax = math.exp(-0.5*math.pow(((ii+0.5-x)/Sx), 2))
    bx = math.exp(-0.5*math.pow((ii-0.5-x)/Sx, 2))

    dudt = - (N/(math.sqrt(2*pi)*Sx*Sx)) * \
        (ax * (ii - x + 0.5) - bx*(ii - x - 0.5)) * PSFy
    d2udt2 = 0

    if (calc_d2):
        d2udt2 = (
            - (2 / Sx)*dudt
            - (N/(math.sqrt(2*pi)*math.pow(Sx, 5)))*(
                ax*math.pow((ii-x+0.5), 3)
                - bx*math.pow((ii-x-0.5), 3)) * PSFy)

    return dudt, d2udt2


@nb.njit(
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int32, nb.int32, nb.float32, nb.float32,
        nb.float32, nb.float32, nb.float32, nb.float32, nb.boolean)
)
def kernel_DerivativeIntGauss2DSigma(ii, jj, x, y, S, N, PSFx, PSFy, calc_d2):
    '''compute the derivative of the 2D gaussian

    Parameters
    ----------
    ii : int
        ???
    jj : int
        ???
    x : float
        ???
    y : float
        ???
    S : float
        ???
    N : float
        ???
    PSFx : float
        ???
    PSFy : float
        ???
    calc_d2 : bool
        calculate d2udt2

    Returns
    -------
    (float, float)
        dudt, d2udt2 values
    '''

    dSx, ddSx = kernel_DerivativeIntGauss1DSigma(
        ii, x, S, N, PSFy, calc_d2)
    dSy, ddSy = kernel_DerivativeIntGauss1DSigma(
        jj, y, S, N, PSFx, calc_d2)

    dudt = dSx + dSy
    d2udt2 = 0
    if (calc_d2):
        d2udt2 = ddSx + ddSy

    return dudt, d2udt2


@nb.njit(
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int64, nb.float32[:]))
def kernel_CenterofMass2D(sz, data):
    '''compute the 2D center of mass of a subregion

    Parameters
    ----------
    sz : _type_
        nxn size of the subregion
    data : _type_
        subregion to search

    Returns
    -------
    tuple[float, float]
        (x, y) coordinate to return
    '''
    tmpx = tmpy = tmpsum = 0.0

    for ii in range(sz):
        for jj in range(sz):
            tmpx += data[sz*jj+ii]*ii
            tmpy += data[sz*jj+ii]*jj
            tmpsum += data[sz*jj+ii]

    x = tmpx / tmpsum
    y = tmpy / tmpsum

    return x, y


@nb.njit(
    nb.types.Tuple((nb.float32, nb.float32))(
        nb.int32, nb.float32, nb.float32[:]))
def kernel_GaussFMaxMin2D(sz, sigma, data):
    '''returns filtered min and pixels of a given subregion

    Parameters
    ----------
    sz : int
        nxn size of the subregion
    sigma : float
        used in filter calculation
    data : float[]
        the subregion to search

    Returns
    -------
    tuple[float, float]
        (MaxN, MinBG) maximum pixel value, minimum background value.
    '''
    # int ii, jj, kk, ll;
    MaxN = 0.0
    MinBG = 10e10  # big

    norm = 1.0 / (2.0 * sigma * sigma)
    # loop over all pixels
    for kk in range(sz):
        for ll in range(sz):
            MaxN = max(MaxN, data[kk*sz+ll])
            MinBG = min(MinBG, data[kk*sz+ll])

    return MaxN, MinBG


@nb.njit(
    nb.types.Tuple((nb.float32[:], nb.float32))(
        nb.int32, nb.int32, nb.float32,
        nb.float32[:], nb.float32[:])
)
def kernel_DerivativeGauss2D(ii, jj, PSFSigma, theta, dudt):
    '''compute DerivativeGauss2D

    Parameters
    ----------
    ii : int
        _description_
    jj : int
        _description_
    PSFSigma : float
        _description_
    theta : float[]
        _description_
    dudt : float[]
        _description_

    Returns
    -------
    tuple[ndarray, float]
        (dudt, model)
    '''
    PSFx = kernel_IntGauss1D(ii, theta[0], PSFSigma)
    PSFy = kernel_IntGauss1D(jj, theta[1], PSFSigma)

    model = theta[3] + theta[2] * PSFx * PSFy

    # calculating derivatives
    dudt[0], _ = kernel_DerivativeIntGauss1D(
        ii, theta[0], PSFSigma, theta[2], PSFy, False)
    dudt[1], _ = kernel_DerivativeIntGauss1D(
        jj, theta[1], PSFSigma, theta[2], PSFx, False)
    dudt[2] = PSFx * PSFy
    dudt[3] = 1.0

    return dudt, model


@nb.njit(
    nb.types.Tuple((nb.float32[:], nb.float32))(
        nb.int32, nb.int32,
        nb.float32[:], nb.float32[:])
)
def kernel_DerivativeGauss2D_sigma(ii, jj, theta, dudt):
    '''compute DerivativeGauss2D_sigma

    Parameters
    ----------
    ii : int
        _description_
    jj : int
        _description_
    PSFSigma : float
        _description_
    theta : float[]
        _description_
    dudt : float[]
        _description_

    Returns
    -------
    tuple[ndarray, float]
        (dudt, model)
    '''
    PSFx = kernel_IntGauss1D(ii, theta[0], theta[4])
    PSFy = kernel_IntGauss1D(jj, theta[1], theta[4])

    model = theta[3] + theta[2] * PSFx * PSFy

    dudt[0], _ = kernel_DerivativeIntGauss1D(
        ii, theta[0], theta[4], theta[2], PSFy, False)
    dudt[1], _ = kernel_DerivativeIntGauss1D(
        jj, theta[1], theta[4], theta[2], PSFx, False)
    dudt[4], _ = kernel_DerivativeIntGauss2DSigma(
        ii, jj,  theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, False)
    dudt[2] = PSFx * PSFy
    dudt[3] = 1.0

    return dudt, model


@nb.njit
def kernel_DerivativeIntGauss2Dz(
        ii, jj, theta, PSFSigma_x, PSFSigma_y, Ax, Ay, Bx, By, gamma, d,
        dudt, d2udt2, calc_d2):
    '''compute the derivative of the 2D gaussian

    Parameters
    ----------
    ii : int
        _description_
    jj : int
        _description_
    theta : float
        _description_
    PSFSigma_x : float
        _description_
    PSFSigma_y : float
        _description_
    Ax : float
        _description_
    Ay : float
        _description_
    Bx : float
        _description_
    By : float
        _description_
    gamma : float
        _description_
    d : float
        _description_
    pPSFx : float
        _description_
    pPSFy : float
        _description_
    dudt : float
        _description_
    d2udt2 : float
        _description_
    calc_d2 : bool
        calculate d2udt2

    Returns
    -------
    tuple[...]
        (PSFx, PSFy, dudt, d2udt2, model)
    '''

    z = theta[4]

    alphax = kernel_alpha(z-gamma, Ax, Bx, d)
    alphay = kernel_alpha(z+gamma, Ay, By, d)

    Sx = PSFSigma_x*math.sqrt(alphax)
    Sy = PSFSigma_y*math.sqrt(alphay)

    PSFx = kernel_IntGauss1D(ii, theta[0], Sx)
    PSFy = kernel_IntGauss1D(jj, theta[1], Sy)
    # pPSFx = PSFx
    # pPSFy = PSFy

    dudt[0], ddx = kernel_DerivativeIntGauss1D(
        ii, theta[0], Sx, theta[2], PSFy, True)
    dudt[1], ddy = kernel_DerivativeIntGauss1D(
        jj, theta[1], Sy, theta[2], PSFx, True)
    dSx, ddSx = kernel_DerivativeIntGauss1DSigma(
        ii, theta[0], Sx, theta[2], PSFy, True)
    dSy, ddSy = kernel_DerivativeIntGauss1DSigma(
        jj, theta[1], Sy, theta[2], PSFx, True)

    dSdalpha_x = PSFSigma_x/(2.0*math.sqrt(alphax))
    dSdalpha_y = PSFSigma_y/(2.0*math.sqrt(alphay))

    dSdzx = dSdalpha_x * kernel_dalphadz(z-gamma, Ax, Bx, d)
    dSdzy = dSdalpha_y * kernel_dalphadz(z+gamma, Ay, By, d)
    dudt[4] = dSx * dSdzx + dSy * dSdzy

    dudt[2] = PSFx * PSFy
    dudt[3] = 1.0

    model = theta[3] + theta[2] * PSFx * PSFy

    if (calc_d2):
        d2udt2[0] = ddx
        d2udt2[1] = ddy

        d2Sdalpha2_x = -PSFSigma_x/(4.0*math.pow(alphax, 1.5))
        d2Sdalpha2_y = -PSFSigma_y/(4.0*math.pow(alphay, 1.5))

        ddSddzx = (
            d2Sdalpha2_x * math.pow(kernel_dalphadz(z-gamma, Ax, Bx, d), 2) +
            dSdalpha_x * kernel_d2alphadz2(z-gamma, Ax, Bx, d))
        ddSddzy = (
            d2Sdalpha2_y*math.pow(kernel_dalphadz(z+gamma, Ay, By, d), 2) +
            dSdalpha_y*kernel_d2alphadz2(z+gamma, Ay, By, d))

        d2udt2[4] = (
            ddSx * (dSdzx * dSdzx) +
            dSx * ddSddzx +
            ddSy * (dSdzy * dSdzy) +
            dSy * ddSddzy)
        d2udt2[2] = 0.0
        d2udt2[3] = 0.0

    return PSFx, PSFy, dudt, d2udt2, model


@nb.njit(
    nb.types.Tuple((nb.float32[:], nb.float32))(
        nb.int32, nb.int32,
        nb.float32[:], nb.float32[:])
)
def kernel_DerivativeGauss2D_sigmaxy(ii, jj, theta, dudt):
    '''
    Parameters
    ----------
    ii : int
        _description_
    jj : int
        _description_
    theta : float[]
        _description_
    dudt : float[]
        _description_

    Returns
    -------
    tuple[ndarray, float]
        (dudt, model)
    '''
    PSFx = kernel_IntGauss1D(ii, theta[0], theta[4])
    PSFy = kernel_IntGauss1D(jj, theta[1], theta[5])

    dudt[0], _ = kernel_DerivativeIntGauss1D(
        ii, theta[0], theta[4], theta[2], PSFy, False)
    dudt[1], _ = kernel_DerivativeIntGauss1D(
        jj, theta[1], theta[5], theta[2], PSFx, False)
    dudt[4], _ = kernel_DerivativeIntGauss1DSigma(
        ii, theta[0], theta[4], theta[2], PSFy, False)
    dudt[5], _ = kernel_DerivativeIntGauss1DSigma(
        jj, theta[1], theta[5], theta[2], PSFx, False)
    dudt[2] = PSFx * PSFy
    dudt[3] = 1.0

    model = theta[3] + theta[2] * PSFx * PSFy

    return dudt, model


@nb.njit()
def kernel_MatInvN(M, Minv, DiagMinv, sz):
    '''nxn partial matrix inversion

    Parameters
    ----------
    M : _type_
        matrix to inverted
    Minv : _type_
        Minv inverted matrix result
    DiagMinv : _type_
        DiagMinv just the inverted diagonal
    sz : _type_
        sz size of the matrix
    '''
    tmp1 = 0.0
    yy = np.zeros(sz, np.float32)

    for jj in range(sz):
        # calculate upper matrix
        for ii in range(jj+1):
            # deal with ii-1 in the sum, set sum(kk=0->ii-1) when ii=0 to zero
            if (ii > 0):
                for kk in range(ii):
                    tmp1 += M[ii+kk*sz]*M[kk+jj*sz]
                M[ii+jj*sz] -= tmp1
                tmp1 = 0

        for ii in range(jj+1, sz, 1):
            if (jj > 0):
                for kk in range(jj):
                    tmp1 += M[ii+kk*sz] * M[kk+jj*sz]
                M[ii+jj*sz] = (1/M[jj+jj*sz])*(M[ii+jj*sz]-tmp1)
                tmp1 = 0
            else:
                M[ii+jj*sz] = (1/M[jj+jj*sz])*M[ii+jj*sz]

    tmp1 = 0

    for num in range(sz):
        # calculate yy
        if (num == 0):
            yy[0] = 1
        else:
            yy[0] = 0

        for ii in range(1, sz, 1):
            if (ii == num):
                b = 1
            else:
                b = 0
            for jj in range(ii):
                tmp1 += M[ii+jj*sz]*yy[jj]
            yy[ii] = b-tmp1
            tmp1 = 0

        # calculate Minv
        Minv[sz-1+num*sz] = yy[sz-1] / M[(sz-1)+(sz-1)*sz]

        for ii in range(sz-2, -1, -1):
            for jj in range(ii+1, sz, 1):
                tmp1 += M[ii+jj*sz]*Minv[jj+num*sz]
            Minv[ii+num*sz] = (1/M[ii+ii*sz])*(yy[ii]-tmp1)
            tmp1 = 0

    # if DiagMinv:
    for ii in range(sz):
        DiagMinv[ii] = Minv[ii * sz + ii]

    return Minv, DiagMinv
