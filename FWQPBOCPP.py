import ctypes
import numpy as np
import sys
import os

IMGTYPE = ctypes.c_float

# Configure the fat-water separation function from the c++ shared library
def init_FWcpp():
    libdir = os.path.join(os.path.dirname(__file__), r'build')
    # if '32 bit' in sys.version:
    #     libfile = 'FW32'
    # else:
    #     libfile = 'FW64'
    libfile = 'libfw'
    try:
        fwlib = np.ctypeslib.load_library(libfile, libdir)
    except:
        print(sys.exc_info())
        raise Exception('{} library not found in dir "{}"'.format(libfile, libdir))

    FWcpp = fwlib.fwqpbo  # Get exported function from DLL
    FWcpp.restype = None  # Needed for void functions

    FWcpp.argtypes = [
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        np.ctypeslib.ndpointer(ctypes.c_float, flags='aligned, contiguous'),
        np.ctypeslib.ndpointer(ctypes.c_float, flags='aligned, contiguous'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_float,
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_int, flags='aligned, contiguous'),
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous'),
        np.ctypeslib.ndpointer(IMGTYPE, flags='aligned, contiguous')]
    return FWcpp


# Perform the actual reconstruction
def reconstruct(dPar, aPar, mPar, B0map=None, R2map=None):
    nVxl = dPar.nx * dPar.ny * dPar.nz

    Yreal = np.real(dPar.img).astype(IMGTYPE)
    Yimag = np.imag(dPar.img).astype(IMGTYPE)

    rhoreal = np.empty(nVxl * mPar.M, dtype=IMGTYPE)
    rhoimag = np.empty(nVxl * mPar.M, dtype=IMGTYPE)

    if B0map is None:
        B0map = np.empty(nVxl, dtype=IMGTYPE)
    if R2map is None:
        R2map = np.empty(nVxl, dtype=IMGTYPE)

    aPar.iR2cand.dtype='int32'

    FWcpp = init_FWcpp()
    FWcpp(Yreal, Yimag, dPar.N, dPar.nx, dPar.ny, dPar.nz, dPar.dx, dPar.dy,
          dPar.dz, dPar.t1, dPar.dt, dPar.B0, mPar.CS, mPar.alpha.flatten(),
          mPar.M, mPar.P, aPar.realEstimates, aPar.R2step, aPar.nR2,
          aPar.iR2cand, aPar.nR2cand, aPar.FibSearch, aPar.mu, aPar.nB0,
          aPar.nICMiter, aPar.maxICMupdate, aPar.graphcutLevel,
          aPar.multiScale, rhoreal, rhoimag, R2map, B0map)

    rho = rhoreal + 1j * rhoimag
    rho.shape = (mPar.M, nVxl)

    return rho, B0map, R2map
