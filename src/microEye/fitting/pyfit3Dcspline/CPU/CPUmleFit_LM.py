import numpy as np
import numba as nb

from .CPUfunctions import *
from .CPUsplineLib import *


@nb.njit
def kernel_MLEFit_LM(
        d_data, PSFSigma, sz, iterations, d_varim=None):
    '''
    brief basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : ndarray
        d_data array of subregions to fit copied to GPU
    PSFSigma : float
        PSFSigma sigma of the point spread function
    sz : int
        sz nxn size of the subregion to fit
    iterations : int
        iterations number of iterations for solution to converge
    d_varim : ndarray
        variance map of CMOS if None then EMCCD
    '''
    NV = NV_P
    M = np.zeros(NV**2, np.float32)
    Diag = np.zeros(NV, np.float32)
    Minv = np.zeros(NV**2, np.float32)

    model = 0
    data = 0
    d_LogLikelihood = Div = 0

    d_Parameters = np.zeros(NV + 1, np.float32)
    d_CRLBs = np.zeros(NV, np.float32)

    newTheta = np.zeros(NV, np.float32)
    oldTheta = np.zeros(NV, np.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = np.zeros(NV, np.float32)
    oldUpdate = np.zeros(NV, np.float32)
    newUpdate[:] = 1e13
    oldUpdate[:] = 1e13

    newDudt = np.zeros(NV, np.float32)
    maxJump = np.array(
        [1.0, 1.0, 100.0, 20.0],
        np.float32)

    newErr = 1e12
    oldErr = 1e13

    jacobian = np.zeros(NV, np.float32)
    hessian = np.zeros(NV**2, np.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = np.zeros(NV**2, np.float32)
    U = np.zeros(NV**2, np.float32)

    # Requires Checking
    s_data = d_data
    s_varim = d_varim

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(sz, PSFSigma, s_data)
    newTheta[2] = max(0.0, (Nmax-newTheta[3])*2*pi*PSFSigma*PSFSigma)
    newTheta[3] = max(newTheta[3], 0.01)

    maxJump[2] = max(newTheta[2], maxJump[2])

    maxJump[3] = max(newTheta[3], maxJump[3])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D(
                ii, jj, PSFSigma, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            if (data > 0):
                newErr = newErr + 2*((model-data)-data*math.log(model/data))
            else:
                newErr = newErr + 2*model
                data = 0

            t1 = 1 - data/model
            for ll in range(NV):
                jacobian[ll] += t1 * newDudt[ll]

            t2 = data/math.pow(model, 2)
            for ll in range(NV):
                for mm in range(ll, NV, 1):
                    hessian[ll*NV + mm] += t2 * newDudt[ll] * newDudt[mm]
                    hessian[mm*NV + ll] = hessian[ll*NV + mm]

    for kk in range(iterations):  # main iterative loop
        if(abs((newErr-oldErr)/newErr) < TOLERANCE):
            break  # CONVERGED
        else:
            if(newErr > ACCEPTANCE * oldErr):
                # copy Fitdata

                for i in range(NV):
                    newTheta[i] = oldTheta[i]
                    newUpdate[i] = oldUpdate[i]
                newLambda = oldLambda
                newErr = oldErr
                mu = max((1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda
            elif(newErr < oldErr and errFlag == 0):
                newLambda = SCALE_DOWN * newLambda
                mu = 1 + newLambda

            for i in range(NV):
                hessian[i*NV + i] = hessian[i*NV + i] * mu

            L[:] = 0
            U[:] = 0

            errFlag = kernel_cholesky(hessian, NV, L, U)
            if (errFlag == 0):
                for i in range(NV):
                    oldTheta[i] = newTheta[i]
                    oldUpdate[i] = newUpdate[i]
                oldLambda = newLambda
                oldErr = newErr

                newUpdate = kernel_luEvaluate(L, U, jacobian, NV, newUpdate)

                # updateFitParameters
                for ll in range(NV):
                    if (newUpdate[ll]/oldUpdate[ll] < -0.5):
                        maxJump[ll] = maxJump[ll] * 0.5

                    newUpdate[ll] = (
                        newUpdate[ll] /
                        (1+math.fabs(newUpdate[ll]/maxJump[ll])))
                    newTheta[ll] = newTheta[ll] - newUpdate[ll]

                # restrict range
                newTheta[0] = max(newTheta[0], (float(sz)-1)/2-sz/4.0)
                newTheta[0] = min(newTheta[0], (float(sz)-1)/2+sz/4.0)
                newTheta[1] = max(newTheta[1], (float(sz)-1)/2-sz/4.0)
                newTheta[1] = min(newTheta[1], (float(sz)-1)/2+sz/4.0)
                newTheta[2] = max(newTheta[2], 1.0)
                newTheta[3] = max(newTheta[3], 0.01)

                newErr = 0
                jacobian[:] = 0
                hessian[:] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D(
                            ii, jj, PSFSigma, newTheta, newDudt)
                        if s_varim is not None:
                            model += s_varim[sz*jj+ii]
                            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
                        else:
                            data = s_data[sz*jj+ii]

                        if (data > 0):
                            newErr = newErr + \
                                2*((model-data)-data*math.log(model/data))
                        else:
                            newErr = newErr + 2 * model
                            data = 0

                        t1 = 1 - data/model
                        for ll in range(NV):
                            jacobian[ll] += t1 * newDudt[ll]

                        t2 = data/math.pow(model, 2)
                        for ll in range(NV):
                            for mm in range(ll, NV, 1):
                                hessian[ll*NV+mm] += \
                                    t2 * newDudt[ll] * newDudt[mm]
                                hessian[mm*NV+ll] = hessian[ll*NV+mm]
            else:
                mu = max(
                    (1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda

    # output iteration

    d_Parameters[NV] = kk
    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D(
                ii, jj, PSFSigma, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            # Building the Fisher Information Matrix
            for kk in range(NV):
                for ll in range(kk, NV, 1):
                    M[kk*NV+ll] += newDudt[ll] * newDudt[kk] / model
                    M[ll*NV+kk] = M[kk*NV+ll]

            # LogLikelyhood
            if (model > 0):
                if (data > 0):
                    Div += \
                        data*math.log(model) - \
                        model-data*math.log(data) + data
                else:
                    Div += - model

    # Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV)
    # Write to global arrays
    for kk in range(NV):
        d_Parameters[kk] = newTheta[kk]
        d_CRLBs[kk] = Diag[kk]
    d_LogLikelihood = Div

    return d_Parameters, d_CRLBs, d_LogLikelihood


@nb.njit
def kernel_MLEFit_LM_Sigma(
        d_data, PSFSigma, sz, iterations, d_varim=None):
    '''
    basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : float[]
        d_data array of subregions to fit copied to GPU
    PSFSigma : float
        PSFSigma sigma of the point spread function
    sz : int
        sz nxn size of the subregion to fit
    iterations : int
        iterations number of iterations for solution to converge
    d_varim : float[]
        variance map of CMOS if None then EMCCD
    '''
    NV = NV_PS

    M = np.zeros(NV**2, np.float32)
    Diag = np.zeros(NV, np.float32)
    Minv = np.zeros(NV**2, np.float32)

    model = 0
    data = 0
    d_LogLikelihood = Div = 0

    d_Parameters = np.zeros(NV + 1, np.float32)
    d_CRLBs = np.zeros(NV, np.float32)

    newTheta = np.zeros(NV, np.float32)
    oldTheta = np.zeros(NV, np.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = np.zeros(NV, np.float32)
    oldUpdate = np.zeros(NV, np.float32)
    newUpdate[:] = 1e13
    oldUpdate[:] = 1e13
    newDudt = np.zeros(NV, np.float32)
    maxJump = np.array(
        [1.0, 1.0, 100.0, 20.0, 0.5],
        np.float32)

    newErr = 1e12
    oldErr = 1e13

    jacobian = np.zeros(NV, np.float32)
    hessian = np.zeros(NV**2, np.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = np.zeros(NV**2, np.float32)
    U = np.zeros(NV**2, np.float32)

    # Requires Checking
    s_data = d_data
    s_varim = d_varim

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(sz, PSFSigma, s_data)
    newTheta[2] = max(0.0, (Nmax-newTheta[3])*2*pi*PSFSigma*PSFSigma)
    newTheta[3] = max(newTheta[3], 0.01)
    newTheta[4] = PSFSigma

    maxJump[2] = max(newTheta[2], maxJump[2])

    maxJump[3] = max(newTheta[3], maxJump[3])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigma(
                ii, jj, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            if (data > 0):
                newErr = newErr + 2*((model-data)-data*math.log(model/data))
            else:
                newErr = newErr + 2*model
                data = 0

            t1 = 1 - data/model
            for ll in range(NV):
                jacobian[ll] += t1 * newDudt[ll]

            t2 = data/math.pow(model, 2)
            for ll in range(NV):
                for mm in range(ll, NV, 1):
                    hessian[ll*NV + mm] += t2 * newDudt[ll] * newDudt[mm]
                    hessian[mm*NV + ll] = hessian[ll*NV + mm]

    for kk in range(iterations):  # main iterative loop
        if(abs((newErr-oldErr)/newErr) < TOLERANCE):
            break  # CONVERGED
        else:
            if(newErr > ACCEPTANCE * oldErr):
                # copy Fitdata

                for i in range(NV):
                    newTheta[i] = oldTheta[i]
                    newUpdate[i] = oldUpdate[i]
                newLambda = oldLambda
                newErr = oldErr
                mu = max((1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda
            elif(newErr < oldErr and errFlag == 0):
                newLambda = SCALE_DOWN * newLambda
                mu = 1 + newLambda

            for i in range(NV):
                hessian[i*NV + i] = hessian[i*NV + i] * mu

            L[:] = 0
            U[:] = 0

            errFlag = kernel_cholesky(hessian, NV, L, U)
            if (errFlag == 0):
                for i in range(NV):
                    oldTheta[i] = newTheta[i]
                    oldUpdate[i] = newUpdate[i]
                oldLambda = newLambda
                oldErr = newErr

                newUpdate = kernel_luEvaluate(L, U, jacobian, NV, newUpdate)

                # updateFitParameters
                for ll in range(NV):
                    if (newUpdate[ll]/oldUpdate[ll] < -0.5):
                        maxJump[ll] = maxJump[ll] * 0.5

                    newUpdate[ll] = (
                        newUpdate[ll] /
                        (1+math.fabs(newUpdate[ll]/maxJump[ll])))
                    newTheta[ll] = newTheta[ll] - newUpdate[ll]
                # restrict range
                newTheta[0] = max(newTheta[0], (float(sz)-1)/2-sz/4.0)
                newTheta[0] = min(newTheta[0], (float(sz)-1)/2+sz/4.0)
                newTheta[1] = max(newTheta[1], (float(sz)-1)/2-sz/4.0)
                newTheta[1] = min(newTheta[1], (float(sz)-1)/2+sz/4.0)
                newTheta[2] = max(newTheta[2], 1.0)
                newTheta[3] = max(newTheta[3], 0.01)
                newTheta[4] = max(newTheta[4], 0.0)
                newTheta[4] = min(newTheta[4], sz/2.0)

                newErr = 0
                jacobian[:] = 0
                hessian[:] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D_sigma(
                            ii, jj, newTheta, newDudt)
                        if s_varim is not None:
                            model += s_varim[sz*jj+ii]
                            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
                        else:
                            data = s_data[sz*jj+ii]

                        if (data > 0):
                            newErr = newErr + \
                                2*((model-data)-data*math.log(model/data))
                        else:
                            newErr = newErr + 2 * model
                            data = 0

                        t1 = 1 - data/model
                        for ll in range(NV):
                            jacobian[ll] += t1 * newDudt[ll]

                        t2 = data/math.pow(model, 2)
                        for ll in range(NV):
                            for mm in range(ll, NV, 1):
                                hessian[ll*NV+mm] += \
                                    t2 * newDudt[ll] * newDudt[mm]
                                hessian[mm*NV+ll] = hessian[ll*NV+mm]
            else:
                mu = max(
                    (1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda

    # output iteration

    d_Parameters[NV] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigma(
                ii, jj, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            # Building the Fisher Information Matrix
            for kk in range(NV):
                for ll in range(kk, NV, 1):
                    M[kk*NV+ll] += newDudt[ll] * newDudt[kk] / model
                    M[ll*NV+kk] = M[kk*NV+ll]

            # LogLikelyhood
            if (model > 0):
                if (data > 0):
                    Div += \
                        data*math.log(model) - \
                        model-data*math.log(data) + data
                else:
                    Div += - model

    # Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV)
    # Write to global arrays
    for kk in range(NV):
        d_Parameters[kk] = newTheta[kk]
        d_CRLBs[kk] = Diag[kk]
    d_LogLikelihood = Div

    return d_Parameters, d_CRLBs, d_LogLikelihood


@nb.njit
def kernel_MLEFit_LM_z(
        d_data, PSFSigma_x, Ax, Ay, Bx, By, gamma, d,
        PSFSigma_y, sz, iterations, d_varim=None):
    '''
    basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : float[]
        array of subregions to fit copied to GPU
    PSFSigma_x : float
        sigma of the point spread function on the x axis
    Ax : float
        ???
    Ay : float
        ???
    Bx : float
        ???
    By : float
        ???
    gamma : float
        ???
    d : float
        ???
    PSFSigma_y : float
        sigma of the point spread function on the y axis
    sz : int
        nxn size of the subregion to fit
    iterations : int
        number of iterations for solution to converge
    d_varim : float[]
        variance map of CMOS if None then EMCCD
    '''
    NV = NV_PS

    M = np.zeros(NV**2, np.float32)
    Diag = np.zeros(NV, np.float32)
    Minv = np.zeros(NV**2, np.float32)

    model = 0
    data = 0
    d_LogLikelihood = Div = 0

    d_Parameters = np.zeros(NV + 1, np.float32)
    d_CRLBs = np.zeros(NV, np.float32)

    # PSFy = PSFx = 0

    newTheta = np.zeros(NV, np.float32)
    oldTheta = np.zeros(NV, np.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = np.zeros(NV, np.float32)
    oldUpdate = np.zeros(NV, np.float32)
    newUpdate[:] = 1e13
    oldUpdate[:] = 1e13
    newDudt = np.zeros(NV, np.float32)
    maxJump = np.array(
        [1.0, 1.0, 100.0, 20.0, 2],
        np.float32)

    newErr = 1e12
    oldErr = 1e13

    jacobian = np.zeros(NV, np.float32)
    hessian = np.zeros(NV**2, np.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = np.zeros(NV**2, np.float32)
    U = np.zeros(NV**2, np.float32)

    # Requires Checking
    s_data = d_data
    s_varim = d_varim

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(
        sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(
        sz, PSFSigma_x, s_data)
    newTheta[2] = max(
        0.0,
        (Nmax-newTheta[3])*2*pi*PSFSigma_x*PSFSigma_y*math.sqrt(2.0))
    newTheta[3] = max(newTheta[3], 0.01)
    newTheta[4] = 0

    maxJump[2] = max(newTheta[2], maxJump[2])

    maxJump[3] = max(newTheta[3], maxJump[3])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            PSFx, PSFy, newDudt, _, model = kernel_DerivativeIntGauss2Dz(
                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            if (data > 0):
                newErr = newErr + 2*((model-data)-data*math.log(model/data))
            else:
                newErr = newErr + 2*model
                data = 0

            t1 = 1 - data/model
            for ll in range(NV):
                jacobian[ll] += t1 * newDudt[ll]

            t2 = data/math.pow(model, 2)
            for ll in range(NV):
                for mm in range(ll, NV, 1):
                    hessian[ll*NV + mm] += t2 * newDudt[ll] * newDudt[mm]
                    hessian[mm*NV + ll] = hessian[ll*NV + mm]

    for kk in range(iterations):  # main iterative loop
        if(abs((newErr-oldErr)/newErr) < TOLERANCE):
            break  # CONVERGED
        else:
            if(newErr > ACCEPTANCE * oldErr):
                # copy Fitdata

                for i in range(NV):
                    newTheta[i] = oldTheta[i]
                    newUpdate[i] = oldUpdate[i]
                newLambda = oldLambda
                newErr = oldErr
                mu = max((1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda
            elif(newErr < oldErr and errFlag == 0):
                newLambda = SCALE_DOWN * newLambda
                mu = 1 + newLambda

            for i in range(NV):
                hessian[i*NV + i] = hessian[i*NV + i] * mu

            L[:] = 0
            U[:] = 0

            errFlag = kernel_cholesky(hessian, NV, L, U)
            if (errFlag == 0):
                for i in range(NV):
                    oldTheta[i] = newTheta[i]
                    oldUpdate[i] = newUpdate[i]
                oldLambda = newLambda
                oldErr = newErr

                newUpdate = kernel_luEvaluate(L, U, jacobian, NV, newUpdate)

                # updateFitParameters
                for ll in range(NV):
                    if (newUpdate[ll]/oldUpdate[ll] < -0.5):
                        maxJump[ll] = maxJump[ll] * 0.5

                    newUpdate[ll] = (
                        newUpdate[ll] /
                        (1+math.fabs(newUpdate[ll]/maxJump[ll])))
                    newTheta[ll] = newTheta[ll] - newUpdate[ll]
                # restrict range
                newTheta[0] = max(newTheta[0], (float(sz)-1)/2-sz/4.0)
                newTheta[0] = min(newTheta[0], (float(sz)-1)/2+sz/4.0)
                newTheta[1] = max(newTheta[1], (float(sz)-1)/2-sz/4.0)
                newTheta[1] = min(newTheta[1], (float(sz)-1)/2+sz/4.0)
                newTheta[2] = max(newTheta[2], 1.0)
                newTheta[3] = max(newTheta[3], 0.01)

                newErr = 0
                jacobian[:] = 0
                hessian[:] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        PSFx, PSFy, newDudt, _, model = \
                            kernel_DerivativeIntGauss2Dz(
                                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)
                        if s_varim is not None:
                            model += s_varim[sz*jj+ii]
                            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
                        else:
                            data = s_data[sz*jj+ii]

                        if (data > 0):
                            newErr = newErr + \
                                2*((model-data)-data*math.log(model/data))
                        else:
                            newErr = newErr + 2 * model
                            data = 0

                        t1 = 1 - data/model
                        for ll in range(NV):
                            jacobian[ll] += t1 * newDudt[ll]

                        t2 = data/math.pow(model, 2)
                        for ll in range(NV):
                            for mm in range(ll, NV, 1):
                                hessian[ll*NV+mm] += \
                                    t2 * newDudt[ll] * newDudt[mm]
                                hessian[mm*NV+ll] = hessian[ll*NV+mm]
            else:
                mu = max(
                    (1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda

    # output iteration

    d_Parameters[NV] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            PSFx, PSFy, newDudt, _, model = kernel_DerivativeIntGauss2Dz(
                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            # Building the Fisher Information Matrix
            for kk in range(NV):
                for ll in range(kk, NV, 1):
                    M[kk*NV+ll] += newDudt[ll] * newDudt[kk] / model
                    M[ll*NV+kk] = M[kk*NV+ll]

            # LogLikelyhood
            if (model > 0):
                if (data > 0):
                    Div += \
                        data*math.log(model) - \
                        model-data*math.log(data) + data
                else:
                    Div += - model

    # Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV)
    # Write to global arrays
    for kk in range(NV):
        d_Parameters[kk] = newTheta[kk]
        d_CRLBs[kk] = Diag[kk]
    d_LogLikelihood = Div

    return d_Parameters, d_CRLBs, d_LogLikelihood


@nb.njit
def kernel_MLEFit_LM_sigmaxy(
        d_data, PSFSigma, sz, iterations, d_varim=None):
    '''
    basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : float[]
        array of subregions to fit copied to GPU
    PSFSigma : float
        sigma of the point spread function
    sz : int
        nxn size of the subregion to fit
    iterations : int
        number of iterations for solution to converge
    d_varim : float[]
        variance map of CMOS if None then EMCCD
    '''
    NV = NV_PS2

    M = np.zeros(NV**2, np.float32)
    Diag = np.zeros(NV, np.float32)
    Minv = np.zeros(NV**2, np.float32)

    model = 0
    data = 0
    d_LogLikelihood = Div = 0

    d_Parameters = np.zeros(NV + 1, np.float32)
    d_CRLBs = np.zeros(NV, np.float32)

    newTheta = np.zeros(NV, np.float32)
    oldTheta = np.zeros(NV, np.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = np.zeros(NV, np.float32)
    oldUpdate = np.zeros(NV, np.float32)
    newUpdate[:] = 1e13
    oldUpdate[:] = 1e13
    newDudt = np.zeros(NV, np.float32)
    maxJump = np.zeros(NV, np.float32)
    maxJump = np.array(
        [1.0, 1.0, 100.0, 20.0, 0.5, 0.5],
        np.float32)

    newErr = 1e12
    oldErr = 1e13

    jacobian = np.zeros(NV, np.float32)
    hessian = np.zeros(NV**2, np.float32)
    t1 = 0.0
    t2 = 0.0

    Nmax = 0
    errFlag = 0
    L = np.zeros(NV**2, np.float32)
    U = np.zeros(NV**2, np.float32)

    # Requires Checking
    s_data = d_data
    s_varim = d_varim

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(
        sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(
        sz, PSFSigma, s_data)
    newTheta[2] = max(
        0.0,
        (Nmax-newTheta[3])*2*pi*PSFSigma*PSFSigma)
    newTheta[3] = max(newTheta[3], 0.01)
    newTheta[4] = PSFSigma
    newTheta[5] = PSFSigma

    maxJump[2] = max(newTheta[2], maxJump[2])

    maxJump[3] = max(newTheta[3], maxJump[3])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                ii,  jj, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            if (data > 0):
                newErr = newErr + 2*((model-data)-data*math.log(model/data))
            else:
                newErr = newErr + 2*model
                data = 0

            t1 = 1 - data/model
            for ll in range(NV):
                jacobian[ll] += t1 * newDudt[ll]

            t2 = data/math.pow(model, 2)
            for ll in range(NV):
                for mm in range(ll, NV, 1):
                    hessian[ll*NV + mm] += t2 * newDudt[ll] * newDudt[mm]
                    hessian[mm*NV + ll] = hessian[ll*NV + mm]

    for kk in range(iterations):  # main iterative loop
        if(abs((newErr-oldErr)/newErr) < TOLERANCE):
            break  # CONVERGED
        else:
            if(newErr > ACCEPTANCE * oldErr):
                # copy Fitdata
                for i in range(NV):
                    newTheta[i] = oldTheta[i]
                    newUpdate[i] = oldUpdate[i]
                newLambda = oldLambda
                newErr = oldErr
                mu = max((1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda
            elif(newErr < oldErr and errFlag == 0):
                newLambda = SCALE_DOWN * newLambda
                mu = 1 + newLambda

            for i in range(NV):
                hessian[i*NV + i] = hessian[i*NV + i] * mu

            L[:] = 0
            U[:] = 0

            errFlag = kernel_cholesky(hessian, NV, L, U)
            if (errFlag == 0):
                for i in range(NV):
                    oldTheta[i] = newTheta[i]
                    oldUpdate[i] = newUpdate[i]
                oldLambda = newLambda
                oldErr = newErr

                newUpdate = kernel_luEvaluate(L, U, jacobian, NV, newUpdate)

                # updateFitParameters
                for ll in range(NV):
                    if (newUpdate[ll]/oldUpdate[ll] < -0.5):
                        maxJump[ll] = maxJump[ll] * 0.5

                    newUpdate[ll] = (
                        newUpdate[ll] /
                        (1+math.fabs(newUpdate[ll]/maxJump[ll])))
                    newTheta[ll] = newTheta[ll] - newUpdate[ll]
                # restrict range
                newTheta[0] = max(newTheta[0], (float(sz)-1)/2-sz/4.0)
                newTheta[0] = min(newTheta[0], (float(sz)-1)/2+sz/4.0)
                newTheta[1] = max(newTheta[1], (float(sz)-1)/2-sz/4.0)
                newTheta[1] = min(newTheta[1], (float(sz)-1)/2+sz/4.0)
                newTheta[2] = max(newTheta[2], 1.0)
                newTheta[3] = max(newTheta[3], 0.01)
                newTheta[4] = max(newTheta[4], PSFSigma/10.0)
                newTheta[5] = max(newTheta[5], PSFSigma/10.0)

                newErr = 0
                jacobian[:] = 0
                hessian[:] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                            ii,  jj, newTheta, newDudt)
                        if s_varim is not None:
                            model += s_varim[sz*jj+ii]
                            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
                        else:
                            data = s_data[sz*jj+ii]

                        if (data > 0):
                            newErr = newErr + \
                                2*((model-data)-data*math.log(model/data))
                        else:
                            newErr = newErr + 2*model
                            data = 0

                        t1 = 1 - data/model
                        for ll in range(NV):
                            jacobian[ll] += t1 * newDudt[ll]

                        t2 = data/math.pow(model, 2)
                        for ll in range(NV):
                            for mm in range(ll, NV, 1):
                                hessian[ll*NV+mm] += \
                                    t2 * newDudt[ll] * newDudt[mm]
                                hessian[mm*NV+ll] = hessian[ll*NV+mm]
            else:
                mu = max(
                    (1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda

    # output iteration

    d_Parameters[NV] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                ii,  jj, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            # Building the Fisher Information Matrix
            for kk in range(NV):
                for ll in range(kk, NV, 1):
                    M[kk*NV+ll] += newDudt[ll] * newDudt[kk] / model
                    M[ll*NV+kk] = M[kk*NV+ll]

            # LogLikelyhood
            if (model > 0):
                if (data > 0):
                    Div += \
                        data*math.log(model) - \
                        model-data*math.log(data) + data
                else:
                    Div += - model

    # Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV)
    # Write to global arrays
    for kk in range(NV):
        d_Parameters[kk] = newTheta[kk]
        d_CRLBs[kk] = Diag[kk]
    d_LogLikelihood = Div

    return d_Parameters, d_CRLBs, d_LogLikelihood


@nb.njit
def kernel_splineMLEFit_z(
        d_data, d_coeff, spline_xsize, spline_ysize, spline_zsize, sz,
        iterations, initZ, d_varim=None):
    '''
    basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : float[]
        array of subregions to fit copied to GPU
    d_coeff : float[]
        array of spline coefficients of the PSF model
    spline_xsize : int
        x size of spline coefficients
    spline_ysize : int
        y size of spline coefficients
    spline_zsize : int
        z size of spline coefficients
    sz : int
        nxn size of the subregion to fit
    iterations : int
        number of iterations for solution to converge
    initZ : float
        intial z position used for fitting
    d_varim : float[]
        variance map of CMOS if None then EMCCD
    '''
    NV = NV_PSP

    M = np.zeros(NV**2, np.float32)
    Diag = np.zeros(NV, np.float32)
    Minv = np.zeros(NV**2, np.float32)

    xstart = ystart = zstart = 0

    s_coeff = d_coeff

    model = 0
    data = 0
    d_LogLikelihood = Div = 0

    d_Parameters = np.zeros(NV + 1, np.float32)
    d_CRLBs = np.zeros(NV, np.float32)

    newTheta = np.zeros(NV, np.float32)
    oldTheta = np.zeros(NV, np.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = np.zeros(NV, np.float32)
    oldUpdate = np.zeros(NV, np.float32)
    newDudt = np.zeros(NV, np.float32)
    newUpdate[:] = 1e13
    oldUpdate[:] = 1e13
    maxJump = np.array(
        [1.0, 1.0, 100.0, 20.0, 2],
        np.float32)

    newErr = 1e12
    oldErr = 1e13

    off = 0.0
    jacobian = np.zeros(NV, np.float32)
    hessian = np.zeros(NV**2, np.float32)
    t1 = 0.0
    t2 = 0.0

    Nmax = 0
    errFlag = 0
    L = np.zeros(NV**2, np.float32)
    U = np.zeros(NV**2, np.float32)

    xc = yc = zc = 0.0
    delta_f = np.zeros(64, np.float32)
    delta_dxf = np.zeros(64, np.float32)
    delta_dyf = np.zeros(64, np.float32)
    delta_dzf = np.zeros(64, np.float32)

    # Requires Checking
    s_data = d_data
    s_varim = d_varim

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(sz, 1.5, s_data)

    # central pixel of spline model
    newTheta[3] = max(newTheta[3], 0.01)
    newTheta[2] = \
        (Nmax-newTheta[3]) / d_coeff[
            int(spline_zsize/2) * (spline_xsize*spline_ysize) +
            int(spline_ysize/2) * spline_xsize +
            int(spline_xsize/2)] * 4

    # newTheta[4]=float(spline_zsize)/2;
    newTheta[4] = initZ

    maxJump[2] = max(newTheta[2], maxJump[2])
    maxJump[3] = max(newTheta[3], maxJump[3])
    maxJump[4] = max(spline_zsize/3.0, maxJump[4])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]

    # updateFitValues
    xc = -1.0*((newTheta[0]-float(sz)/2)+0.5)
    yc = -1.0*((newTheta[1]-float(sz)/2)+0.5)

    off = math.floor((float(spline_xsize)+1.0-float(sz))/2)

    xstart = math.floor(xc)
    xc = xc - xstart

    ystart = math.floor(yc)
    yc = yc - ystart

    # zstart = floor(newTheta[4]);
    zstart = math.floor(newTheta[4])
    zc = newTheta[4] - zstart

    # updateFitValues
    newErr = 0
    delta_f, delta_dxf, delta_dyf, delta_dzf = kernel_computeDelta3D(
        xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)

    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeSpline(
                ii + xstart + off, jj + ystart + off, zstart,
                spline_xsize, spline_ysize, spline_zsize,
                delta_f, delta_dxf, delta_dyf, delta_dzf,
                s_coeff, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            if (data > 0):
                newErr = newErr + 2*((model-data)-data*math.log(model/data))
            else:
                newErr = newErr + 2*model
                data = 0

            t1 = 1 - data/model
            for ll in range(NV):
                jacobian[ll] += t1 * newDudt[ll]

            t2 = data/math.pow(model, 2)
            for ll in range(NV):
                for mm in range(ll, NV, 1):
                    hessian[ll*NV + mm] += t2 * newDudt[ll] * newDudt[mm]
                    hessian[mm*NV + ll] = hessian[ll*NV + mm]

    for kk in range(iterations):  # main iterative loop
        if(abs((newErr-oldErr)/newErr) < TOLERANCE):
            break  # CONVERGED
        else:
            if(newErr > ACCEPTANCE * oldErr):
                # copy Fitdata
                for i in range(NV):
                    newTheta[i] = oldTheta[i]
                    newUpdate[i] = oldUpdate[i]
                newLambda = oldLambda
                newErr = oldErr
                mu = max((1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda
            elif(newErr < oldErr and errFlag == 0):
                newLambda = SCALE_DOWN * newLambda
                mu = 1 + newLambda

            for i in range(NV):
                hessian[i*NV + i] = hessian[i*NV + i] * mu

            L[:] = 0
            U[:] = 0

            errFlag = kernel_cholesky(hessian, NV, L, U)

            if (errFlag == 0):
                for i in range(NV):
                    oldTheta[i] = newTheta[i]
                    oldUpdate[i] = newUpdate[i]
                oldLambda = newLambda
                oldErr = newErr

                newUpdate = kernel_luEvaluate(L, U, jacobian, NV, newUpdate)

                # updateFitParameters
                for ll in range(NV):
                    if (newUpdate[ll]/oldUpdate[ll] < -0.5):
                        maxJump[ll] = maxJump[ll] * 0.5

                    newUpdate[ll] = (
                        newUpdate[ll] /
                        (1+math.fabs(newUpdate[ll]/maxJump[ll])))
                    newTheta[ll] = newTheta[ll] - newUpdate[ll]
                # restrict range
                newTheta[0] = max(newTheta[0], (float(sz)-1)/2-sz/4.0)
                newTheta[0] = min(newTheta[0], (float(sz)-1)/2+sz/4.0)
                newTheta[1] = max(newTheta[1], (float(sz)-1)/2-sz/4.0)
                newTheta[1] = min(newTheta[1], (float(sz)-1)/2+sz/4.0)
                newTheta[2] = max(newTheta[2], 1.0)
                newTheta[3] = max(newTheta[3], 0.01)
                newTheta[4] = max(newTheta[4], 0.0)
                newTheta[4] = min(newTheta[4], float(spline_zsize))

                # updateFitValues
                xc = -1.0*((newTheta[0]-float(sz)/2)+0.5)
                yc = -1.0*((newTheta[1]-float(sz)/2)+0.5)

                xstart = math.floor(xc)
                xc = xc - xstart

                ystart = math.floor(yc)
                yc = yc - ystart

                zstart = math.floor(newTheta[4])
                zc = newTheta[4] - zstart

                newErr = 0
                jacobian[:] = 0
                hessian[:] = 0
                delta_f, delta_dxf, delta_dyf, delta_dzf = \
                    kernel_computeDelta3D(
                        xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)

                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeSpline(
                            ii + xstart + off, jj + ystart + off, zstart,
                            spline_xsize, spline_ysize, spline_zsize,
                            delta_f, delta_dxf, delta_dyf, delta_dzf,
                            s_coeff, newTheta, newDudt)
                        if s_varim is not None:
                            model += s_varim[sz*jj+ii]
                            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
                        else:
                            data = s_data[sz*jj+ii]

                        if (data > 0):
                            newErr = newErr + \
                                2*((model-data)-data*math.log(model/data))
                        else:
                            newErr = newErr + 2*model
                            data = 0

                        t1 = 1 - data/model
                        for ll in range(NV):
                            jacobian[ll] += t1 * newDudt[ll]

                        t2 = data/math.pow(model, 2)
                        for ll in range(NV):
                            for mm in range(ll, NV, 1):
                                hessian[ll*NV+mm] += \
                                    t2 * newDudt[ll] * newDudt[mm]
                                hessian[mm*NV+ll] = hessian[ll*NV+mm]
            else:
                mu = max(
                    (1 + newLambda*SCALE_UP)/(1 + newLambda), 1.3)
                newLambda = SCALE_UP * newLambda

    # output iteration

    d_Parameters[NV] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0

    xc = -1.0*((newTheta[0]-float(sz)/2)+0.5)
    yc = -1.0*((newTheta[1]-float(sz)/2)+0.5)

    # off = (float(spline_xsize)+1.0-2*float(sz))/2;

    xstart = math.floor(xc)
    xc = xc - xstart

    ystart = math.floor(yc)
    yc = yc - ystart

    zstart = math.floor(newTheta[4])
    zc = newTheta[4] - zstart

    delta_f, delta_dxf, delta_dyf, delta_dzf = kernel_computeDelta3D(
        xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)

    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeSpline(
                ii + xstart + off, jj + ystart + off, zstart,
                spline_xsize, spline_ysize, spline_zsize,
                delta_f, delta_dxf, delta_dyf, delta_dzf,
                s_coeff, newTheta, newDudt)
            if s_varim is not None:
                model += s_varim[sz*jj+ii]
                data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]
            else:
                data = s_data[sz*jj+ii]

            # Building the Fisher Information Matrix
            for kk in range(NV):
                for ll in range(kk, NV, 1):
                    M[kk*NV+ll] += newDudt[ll] * newDudt[kk] / model
                    M[ll*NV+kk] = M[kk*NV+ll]

            # LogLikelyhood
            if (model > 0):
                if (data > 0):
                    Div += \
                        data*math.log(model) - \
                        model-data*math.log(data) + data
                else:
                    Div += - model

    # Matrix inverse (CRLB=F^-1) and output assigments
    kernel_MatInvN(M, Minv, Diag, NV)
    # Write to global arrays
    for kk in range(NV):
        d_Parameters[kk] = newTheta[kk]
        d_CRLBs[kk] = Diag[kk]
    d_LogLikelihood = Div

    return d_Parameters, d_CRLBs, d_LogLikelihood
