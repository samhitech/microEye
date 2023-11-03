from numba import cuda
from .mainfunctions import CPUmleFit_LM, \
    get_roi_list, get_roi_list_CMOS

if cuda.is_available():
    from .mainfunctions import GPUmleFit_LM
