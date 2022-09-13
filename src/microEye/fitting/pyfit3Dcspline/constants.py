import numpy as np

BSZ = 128		# number of threads per block
MEM = 3872      # 3872		11616# !< shared
IMSZBIG = 51 	# !< maximum fitting window size
NK = 128		# !< number of blocks to run in each kernel
pi = np.pi 	# !< ensure a consistent value for pi
NV_P = 4		# !< number of fitting parameters for MLEfit (x,y,bg,I)
NV_PS = 5		# !< number of fitting parameters for MLEFit_sigma (x,y,bg,I,Sigma)
NV_PZ = 5		# !< not used (x,y,bg,I,z)
NV_PS2 = 6		# !< number of parameters for MLEFit_sigmaxy (x,y,bg,I,Sx,Sy)
NV_PSP = 5

NV_P_squared = NV_P**2
NV_PS_squared = NV_PS**2
NV_PZ_squared = NV_PZ**2
NV_PS2_squared = NV_PS2**2
NV_PSP_squared = NV_PSP**2

TOLERANCE = 1e-6
INIT_ERR = 1e13
INIT_LAMBDA = 0.1
SCALE_UP = 10
SCALE_DOWN = 0.1
ACCEPTANCE = 1.5

BLOCK_MAX_SIZE = 512
