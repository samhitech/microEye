from numba import cuda

from microEye.analysis.fitting.pyfit3Dcspline.mainfunctions import (
    CPUmleFit_LM,
    get_roi_list,
    get_roi_list_CMOS,
)

if cuda.is_available():
    from microEye.analysis.fitting.pyfit3Dcspline.mainfunctions import GPUmleFit_LM
