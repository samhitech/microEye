
import numba as nb
from numba import cuda

from .GPUfunctions import *
from .GPUsplineLib import *

from ..constants import *


@cuda.jit
def kernel_MLEFit_LM_sCMOS(
        d_data, PSFSigma, sz, iterations, d_Parameters, d_CRLBs,
        d_LogLikelihood, Nfits, d_varim):
    '''
    brief basic MLE fitting kernel.  No additional parameters are computed.

    Parameters
    ----------
    d_data : _type_
        d_data array of subregions to fit copied to GPU
    PSFSigma : _type_
        PSFSigma sigma of the point spread function
    sz : _type_
        sz nxn size of the subregion to fit
    iterations : _type_
        iterations number of iterations for solution to converge
    d_Parameters : _type_
        d_Parameters array of fitting parameters to return for each subregion
    d_CRLBs : _type_
        d_CRLBs array of Cramer-Rao lower bound estimates to return for
        each subregion
    d_LogLikelihood : _type_
        d_LogLikelihood array of loglikelihood estimates to return for
        each subregion
    Nfits : _type_
        Nfits number of subregions to fit
    d_varim : _type_
        variance map of scmos
    '''
    NV = NV_P
    NV_NV = NV_P_squared

    M = cuda.local.array(NV_NV, nb.float32)
    Diag = cuda.local.array(NV, nb.float32)
    Minv = cuda.local.array(NV_NV, nb.float32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    BlockSize = cuda.blockDim.x

    model = 0
    data = 0
    Div = 0

    newTheta = cuda.local.array(NV, nb.float32)
    oldTheta = cuda.local.array(NV, nb.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = cuda.local.array(NV, nb.float32)
    oldUpdate = cuda.local.array(NV, nb.float32)
    newDudt = cuda.local.array(NV, nb.float32)
    for i in range(NV):
        newUpdate[i] = 1e13
        oldUpdate[i] = 1e13
        newDudt[i] = 0
    maxJump = cuda.local.array(4, nb.float32)
    maxJump[0] = 1.0
    maxJump[1] = 1.0
    maxJump[2] = 100.0
    maxJump[3] = 20.0

    newErr = 1e12
    oldErr = 1e13

    jacobian = cuda.local.array(NV, nb.float32)
    hessian = cuda.local.array(NV_NV, nb.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = cuda.local.array(NV_NV, nb.float32)
    U = cuda.local.array(NV_NV, nb.float32)

    # Prevent read/write past end of array
    if ((bx*BlockSize+tx) >= Nfits):
        return

    for ii in range(NV_NV):
        M[ii] = 0
        Minv[ii] = 0
        hessian[ii] = 0
        L[ii] = 0
        U[ii] = 0

    # Requires Checking
    idx = sz*sz*bx*BlockSize + sz*sz*tx
    s_data = d_data[idx:idx + sz*sz]
    s_varim = d_varim[idx:idx + sz*sz]

    # initial values
    newTheta[0], newTheta[1] = kernel_CenterofMass2D(sz, s_data)
    Nmax, newTheta[3] = kernel_GaussFMaxMin2D(sz, PSFSigma, s_data)
    newTheta[2] = max(0.0, (Nmax-newTheta[3])*2*pi*PSFSigma*PSFSigma)
    newTheta[3] = max(newTheta[3], 0.01)

    maxJump[2] = max(newTheta[2], maxJump[2])

    maxJump[3] = max(newTheta[3], maxJump[3])

    for ii in range(NV):
        oldTheta[ii] = newTheta[ii]
        jacobian[ii] = 0

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D(
                ii, jj, PSFSigma, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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

            for ii in range(NV_NV):
                L[ii] = 0
                U[ii] = 0

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
                for hh in range(NV):
                    jacobian[hh] = 0
                for hh in range(NV_NV):
                    hessian[hh] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D(
                            ii, jj, PSFSigma, newTheta, newDudt)
                        model += s_varim[sz*jj+ii]
                        data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
    d_Parameters[(NV + 1)*(1+BlockSize*bx+tx) - 1] = kk
    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D(
                ii, jj, PSFSigma, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
        d_Parameters[kk + (NV + 1)*(BlockSize*bx+tx)] = newTheta[kk]
        d_CRLBs[kk + NV*(BlockSize*bx+tx)] = Diag[kk]
    d_LogLikelihood[BlockSize*bx+tx] = Div

    return


@cuda.jit
def kernel_MLEFit_LM_Sigma_sCMOS(
        d_data, PSFSigma, sz, iterations, d_Parameters,
        d_CRLBs, d_LogLikelihood, Nfits, d_varim):
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
    d_Parameters : float[]
        d_Parameters array of fitting parameters to return for each subregion
    d_CRLBs : float[]
        d_CRLBs array of Cramer-Rao lower bound estimates to return for
        each subregion
    d_LogLikelihood : float[]
        d_LogLikelihood array of loglikelihood estimates to return for
        each subregion
    Nfits : int
        Nfits number of subregions to fit
    d_varim : float[]
        variance map of scmos
    '''
    NV = NV_PS
    NV_NV = NV_PS_squared

    M = cuda.local.array(NV_NV, nb.float32)
    Diag = cuda.local.array(NV, nb.float32)
    Minv = cuda.local.array(NV_NV, nb.float32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    BlockSize = cuda.blockDim.x

    model = 0
    data = 0
    Div = 0

    newTheta = cuda.local.array(NV, nb.float32)
    oldTheta = cuda.local.array(NV, nb.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = cuda.local.array(NV, nb.float32)
    oldUpdate = cuda.local.array(NV, nb.float32)
    newDudt = cuda.local.array(NV, nb.float32)
    for i in range(NV):
        newUpdate[i] = 1e13
        oldUpdate[i] = 1e13
        newDudt[i] = 0
    maxJump = cuda.local.array(NV, nb.float32)
    maxJump[0] = 1.0
    maxJump[1] = 1.0
    maxJump[2] = 100.0
    maxJump[3] = 20.0
    maxJump[4] = 0.5

    newErr = 1e12
    oldErr = 1e13

    jacobian = cuda.local.array(NV, nb.float32)
    hessian = cuda.local.array(NV_NV, nb.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = cuda.local.array(NV_NV, nb.float32)
    U = cuda.local.array(NV_NV, nb.float32)

    #  Prevent read/write past end of array
    if ((bx*BlockSize+tx) >= Nfits):
        return

    for ii in range(NV_NV):
        M[ii] = 0
        Minv[ii] = 0
        hessian[ii] = 0
        L[ii] = 0
        U[ii] = 0

    # Requires Checking
    idx = sz*sz*bx*BlockSize + sz*sz*tx
    s_data = d_data[idx:idx + sz*sz]
    s_varim = d_varim[idx:idx + sz*sz]

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
        jacobian[ii] = 0

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigma(
                ii, jj, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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

            for ii in range(NV_NV):
                L[ii] = 0
                U[ii] = 0

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
                for hh in range(NV):
                    jacobian[hh] = 0
                for hh in range(NV_NV):
                    hessian[hh] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D_sigma(
                            ii, jj, newTheta, newDudt)
                        model += s_varim[sz*jj+ii]
                        data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
    d_Parameters[(NV + 1)*(1+BlockSize*bx+tx) - 1] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigma(
                ii, jj, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
        d_Parameters[kk + (NV + 1)*(BlockSize*bx+tx)] = newTheta[kk]
        d_CRLBs[kk + NV*(BlockSize*bx+tx)] = Diag[kk]
    d_LogLikelihood[BlockSize*bx+tx] = Div

    return


@cuda.jit
def kernel_MLEFit_LM_z_sCMOS(
        d_data, PSFSigma_x, Ax, Ay, Bx, By, gamma, d,
        PSFSigma_y, sz, iterations, d_Parameters, d_CRLBs,
        d_LogLikelihood, Nfits, d_varim):
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
    d_Parameters : float[]
        array of fitting parameters to return for each subregion
    d_CRLBs : float[]
        array of Cramer-Rao lower bound estimates to return for each subregion
    d_LogLikelihood : float[]
        array of loglikelihood estimates to return for each subregion
    Nfits : int
        number of subregions to fit
    d_varim : float[]
        variance map of scmos
    '''
    NV = NV_PS
    NV_NV = NV_PS_squared

    M = cuda.local.array(NV_NV, nb.float32)
    Diag = cuda.local.array(NV, nb.float32)
    Minv = cuda.local.array(NV_NV, nb.float32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    BlockSize = cuda.blockDim.x

    model = 0
    data = 0
    Div = 0
    PSFy = PSFx = 0

    newTheta = cuda.local.array(NV, nb.float32)
    oldTheta = cuda.local.array(NV, nb.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = cuda.local.array(NV, nb.float32)
    oldUpdate = cuda.local.array(NV, nb.float32)
    newDudt = cuda.local.array(NV, nb.float32)
    for i in range(NV):
        newUpdate[i] = 1e13
        oldUpdate[i] = 1e13
        newDudt[i] = 0
    maxJump = cuda.local.array(NV, nb.float32)
    maxJump[0] = 1.0
    maxJump[1] = 1.0
    maxJump[2] = 100.0
    maxJump[3] = 20.0
    maxJump[4] = 2

    newErr = 1e12
    oldErr = 1e13

    jacobian = cuda.local.array(NV, nb.float32)
    hessian = cuda.local.array(NV_NV, nb.float32)
    t1 = 0
    t2 = 0

    Nmax = 0
    errFlag = 0
    L = cuda.local.array(NV_NV, nb.float32)
    U = cuda.local.array(NV_NV, nb.float32)

    #  Prevent read/write past end of array
    if ((bx*BlockSize+tx) >= Nfits):
        return

    for ii in range(NV_NV):
        M[ii] = 0
        Minv[ii] = 0
        hessian[ii] = 0
        L[ii] = 0
        U[ii] = 0

    # Requires Checking
    idx = sz*sz*bx*BlockSize + sz*sz*tx
    s_data = d_data[idx:idx + sz*sz]
    s_varim = d_varim[idx:idx + sz*sz]

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
        jacobian[ii] = 0

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            PSFx, PSFy, newDudt, _, model = kernel_DerivativeIntGauss2Dz(
                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii]+s_varim[sz*jj+ii]

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

            for ii in range(NV_NV):
                L[ii] = 0
                U[ii] = 0

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
                for hh in range(NV):
                    jacobian[hh] = 0
                for hh in range(NV_NV):
                    hessian[hh] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        PSFx, PSFy, newDudt, _, model = \
                            kernel_DerivativeIntGauss2Dz(
                                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)

                        model += s_varim[sz*jj+ii]
                        data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
    d_Parameters[(NV + 1)*(1+BlockSize*bx+tx) - 1] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            PSFx, PSFy, newDudt, _, model = kernel_DerivativeIntGauss2Dz(
                ii, jj, newTheta, PSFSigma_x, PSFSigma_y,
                Ax, Ay, Bx, By, gamma, d, newDudt, None, False)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
        d_Parameters[kk + (NV + 1)*(BlockSize*bx+tx)] = newTheta[kk]
        d_CRLBs[kk + NV*(BlockSize*bx+tx)] = Diag[kk]
    d_LogLikelihood[BlockSize*bx+tx] = Div

    return


@cuda.jit
def kernel_MLEFit_LM_sigmaxy_sCMOS(
        d_data, PSFSigma, sz, iterations, d_Parameters,
        d_CRLBs, d_LogLikelihood, Nfits, d_varim):
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
    d_Parameters : float[]
        array of fitting parameters to return for each subregion
    d_CRLBs : float[]
        array of Cramer-Rao lower bound estimates to return for each subregion
    d_LogLikelihood : float[]
        array of loglikelihood estimates to return for each subregion
    Nfits : int
        number of subregions to fit
    d_varim : float[]
        variance map of scmos
    '''
    NV = NV_PS2
    NV_NV = NV_PS2_squared

    M = cuda.local.array(NV_NV, nb.float32)
    Diag = cuda.local.array(NV, nb.float32)
    Minv = cuda.local.array(NV_NV, nb.float32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    BlockSize = cuda.blockDim.x

    model = 0
    data = 0
    Div = 0

    newTheta = cuda.local.array(NV, nb.float32)
    oldTheta = cuda.local.array(NV, nb.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = cuda.local.array(NV, nb.float32)
    oldUpdate = cuda.local.array(NV, nb.float32)
    newDudt = cuda.local.array(NV, nb.float32)
    for i in range(NV):
        newUpdate[i] = 1e13
        oldUpdate[i] = 1e13
        newDudt[i] = 0
    maxJump = cuda.local.array(NV, nb.float32)
    maxJump[0] = 1.0
    maxJump[1] = 1.0
    maxJump[2] = 100.0
    maxJump[3] = 20.0
    maxJump[4] = 0.5
    maxJump[5] = 0.5

    newErr = 1e12
    oldErr = 1e13

    jacobian = cuda.local.array(NV, nb.float32)
    hessian = cuda.local.array(NV_NV, nb.float32)
    t1 = 0.0
    t2 = 0.0

    Nmax = 0
    errFlag = 0
    L = cuda.local.array(NV_NV, nb.float32)
    U = cuda.local.array(NV_NV, nb.float32)

    #  Prevent read/write past end of array
    if ((bx*BlockSize+tx) >= Nfits):
        return

    for ii in range(NV_NV):
        M[ii] = 0
        Minv[ii] = 0
        hessian[ii] = 0
        L[ii] = 0
        U[ii] = 0

    # Requires Checking
    idx = sz*sz*bx*BlockSize + sz*sz*tx
    s_data = d_data[idx:idx + sz*sz]
    s_varim = d_varim[idx:idx + sz*sz]

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
        jacobian[ii] = 0

    # updateFitValues
    newErr = 0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                ii,  jj, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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

            for ii in range(NV_NV):
                L[ii] = 0
                U[ii] = 0

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
                for hh in range(NV):
                    jacobian[hh] = 0
                for hh in range(NV_NV):
                    hessian[hh] = 0
                for ii in range(sz):
                    for jj in range(sz):
                        # calculating derivatives
                        newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                            ii,  jj, newTheta, newDudt)
                        model += s_varim[sz*jj+ii]
                        data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
    d_Parameters[(NV + 1)*(1+BlockSize*bx+tx) - 1] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    for ii in range(sz):
        for jj in range(sz):
            newDudt, model = kernel_DerivativeGauss2D_sigmaxy(
                ii,  jj, newTheta, newDudt)
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii]+s_varim[sz*jj+ii]

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
        d_Parameters[kk + (NV + 1)*(BlockSize*bx+tx)] = newTheta[kk]
        d_CRLBs[kk + NV*(BlockSize*bx+tx)] = Diag[kk]
    d_LogLikelihood[BlockSize*bx+tx] = Div

    return


@cuda.jit
def kernel_splineMLEFit_z_sCMOS(
        d_data, d_coeff, spline_xsize, spline_ysize, spline_zsize, sz,
        iterations, d_Parameters, d_CRLBs, d_LogLikelihood,
        initZ, Nfits, d_varim):
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
    d_Parameters : float[]
        array of fitting parameters to return for each subregion
    d_CRLBs : float[]
        array of Cramer-Rao lower bound estimates to return for each subregion
    d_LogLikelihood : float[]
        array of loglikelihood estimates to return for each subregion
    initZ : float
        intial z position used for fitting
    Nfits : int
        number of subregions to fit
    d_varim : float[]
        variance map of sCMOS
    '''
    NV = NV_PSP
    NV_NV = NV_PSP_squared

    M = cuda.local.array(NV_NV, nb.float32)
    Diag = cuda.local.array(NV, nb.float32)
    Minv = cuda.local.array(NV_NV, nb.float32)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    BlockSize = cuda.blockDim.x

    xstart = ystart = zstart = 0

    s_coeff = d_coeff

    model = 0
    data = 0
    Div = 0

    newTheta = cuda.local.array(NV, nb.float32)
    oldTheta = cuda.local.array(NV, nb.float32)
    newLambda = INIT_LAMBDA
    oldLambda = INIT_LAMBDA
    mu = 0

    newUpdate = cuda.local.array(NV, nb.float32)
    oldUpdate = cuda.local.array(NV, nb.float32)
    newDudt = cuda.local.array(NV, nb.float32)
    for i in range(NV):
        newUpdate[i] = 1e13
        oldUpdate[i] = 1e13
        newDudt[i] = 0
    maxJump = cuda.local.array(NV, nb.float32)
    maxJump[0] = 1.0
    maxJump[1] = 1.0
    maxJump[2] = 100.0
    maxJump[3] = 20.0
    maxJump[4] = 2

    newErr = 1e12
    oldErr = 1e13

    off = 0.0
    jacobian = cuda.local.array(NV, nb.float32)
    hessian = cuda.local.array(NV_NV, nb.float32)
    t1 = 0.0
    t2 = 0.0

    Nmax = 0
    errFlag = 0
    L = cuda.local.array(NV_NV, nb.float32)
    U = cuda.local.array(NV_NV, nb.float32)

    xc = yc = zc = 0.0
    delta_f = cuda.local.array(64, nb.float32)
    delta_dxf = cuda.local.array(64, nb.float32)
    delta_dyf = cuda.local.array(64, nb.float32)
    delta_dzf = cuda.local.array(64, nb.float32)
    for ii in range(64):
        delta_f[ii] = 0
        delta_dxf[ii] = 0
        delta_dyf[ii] = 0
        delta_dzf[ii] = 0

    #  Prevent read/write past end of array
    if ((bx*BlockSize+tx) >= Nfits):
        return

    for ii in range(NV_NV):
        M[ii] = 0
        Minv[ii] = 0
        hessian[ii] = 0
        L[ii] = 0
        U[ii] = 0

    # Requires Checking
    idx = sz*sz*bx*BlockSize + sz*sz*tx
    s_data = d_data[idx:idx + sz*sz]
    s_varim = d_varim[idx:idx + sz*sz]

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
        jacobian[ii] = 0

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
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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

            for ii in range(NV_NV):
                L[ii] = 0
                U[ii] = 0

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
                for hh in range(NV):
                    jacobian[hh] = 0
                for hh in range(NV_NV):
                    hessian[hh] = 0
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
                        model += s_varim[sz*jj+ii]
                        data = s_data[sz*jj+ii] + s_varim[sz*jj+ii]

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
    d_Parameters[(NV + 1)*(1+BlockSize*bx+tx) - 1] = kk

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
            model += s_varim[sz*jj+ii]
            data = s_data[sz*jj+ii]+s_varim[sz*jj+ii]

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
        d_Parameters[kk + (NV + 1)*(BlockSize*bx+tx)] = newTheta[kk]
        d_CRLBs[kk + NV*(BlockSize*bx+tx)] = Diag[kk]
    d_LogLikelihood[BlockSize*bx+tx] = Div

    return
