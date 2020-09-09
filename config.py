import configparser
import DICOM
import MATLAB
import numpy as np
import os


# Helper class for convenient reading of config files
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# extract data parameter object representing a single slice
def getSliceDataParams(dPar, slice, z):
    sliceDataParams = AttrDict(dPar)
    sliceDataParams.sliceList = [slice]
    sliceDataParams.img = dPar.img.reshape(
        dPar.N, dPar.nz, dPar.ny*dPar.nx)[:, z, :].flatten()
    sliceDataParams.nz = 1
    return sliceDataParams


# extract dPar object representing a slab of contiguous slices starting at z
def getSlabDataParams(dPar, slices, z):
    slabDataParams = AttrDict(dPar)
    slabDataParams.sliceList = slices
    slabSize = len(slices)
    slabDataParams.img = dPar.img.reshape(
        dPar.N, dPar.nz, dPar.ny*dPar.nx)[:, z:z+slabSize, :].flatten()
    slabDataParams.nz = slabSize
    return slabDataParams


# Update algorithm parameter object aPar and set default parameters
def setupAlgoParams(aPar, N, nFAC=0):
    if 'nr2' in aPar:
        aPar.nR2 = int(aPar.nr2)
    else:
        aPar.nR2 = 1
    if 'r2max' in aPar:
        aPar.R2max = float(aPar.r2max)
    else:
        aPar.R2max = 100.
    if 'r2cand' in aPar:
        aPar.R2cand = [float(R2) for R2 in aPar.r2cand.split(',')]
    else:
        aPar.R2cand = [0.]
    if 'fibsearch' in aPar:
        aPar.FibSearch = aPar.fibsearch == 'True'
    else:
        aPar.FibSearch = False
    if 'mu' in aPar:
        aPar.mu = float(aPar.mu)
    else:
        aPar.mu = 1.
    if 'nb0' in aPar:
        aPar.nB0 = int(aPar.nb0)
    else:
        aPar.nB0 = 100
    if 'nicmiter' in aPar:
        aPar.nICMiter = int(aPar.nicmiter)
    else:
        aPar.nICMiter = 0
    if 'graphcut' in aPar:
        aPar.graphcut = aPar.graphcut == 'True'
    else:
        aPar.graphcut = 'graphcutlevel' in aPar
    if aPar.graphcut:
        if 'graphcutlevel' in aPar:
            aPar.graphcutLevel = int(aPar.graphcutlevel)
        else:
            aPar.graphcutLevel = 0
    else:
        aPar.graphcutLevel = None
    if 'multiscale' in aPar:
        aPar.multiScale = aPar.multiscale == 'True'
    else:
        aPar.multiScale = False
    if 'use3d' in aPar:
        aPar.use3D = aPar.use3d == 'True'
    else:
        aPar.use3D = False
    if 'magnitudediscrimination' in aPar:
        aPar.magnDiscr = aPar.magnitudediscrimination == 'True'
    else:
        aPar.magnDiscr = True
    if 'realestimates' in aPar:
        aPar.realEstimates = aPar.realestimates == 'True'
        if not aPar.realEstimates and N == 2:
            raise Exception('Real-valued estimates needed for two-point Dixon')
    elif N == 2:
        aPar.realEstimates = True
    else:
        aPar.realEstimates = False
    if 'offrespenalty' in aPar:
        aPar.offresPenalty = float(aPar.offrespenalty)
    else:
        aPar.offresPenalty = 0

    if aPar.nR2 > 1:
        aPar.R2step = aPar.R2max/(aPar.nR2-1)  # [sec-1]
    else:
        aPar.R2step = 1.0  # [sec-1]
    aPar.iR2cand = np.array(list(set([min(aPar.nR2-1, int(R2/aPar.R2step))
                            for R2 in aPar.R2cand])))  # [msec]
    aPar.nR2cand = len(aPar.iR2cand)
    aPar.maxICMupdate = round(aPar.nB0/10)

    # For Fatty Acid Composition, create algorithmParams for two passes: aPar and aPar.pass2
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: use B0- and R2*-maps from first pass
    if nFAC > 0:
        aPar.pass2 = AttrDict(aPar)  # modify algoParams for pass 2:
        aPar.pass2.nICMiter = 0  # to omit ICM
        aPar.pass2.graphcutLevel = None  # to omit the graphcut
        aPar.pass2.graphcut = False
    
    aPar.output = ['wat', 'fat', 'ff', 'B0map']
    if aPar.realEstimates:
        aPar.output.append('phi')
    if (aPar.nR2 > 1):
        aPar.output.append('R2map')
    if (nFAC > 2):
        aPar.output.append('CL')
    if (nFAC > 1):
        aPar.output.append('PUD')
    if (nFAC > 0):
        aPar.output.append('UD')


# Get relative weights alpha of fat resonances based on CL, UD, and PUD per UD
def getFACalphas(CL=None, P2U=None, UD=None):
    P = 11  # Expects one water and ten triglyceride resonances
    M = [CL, UD, P2U].count(None)+2
    alpha = np.zeros([M, P], dtype=np.float32)
    alpha[0, 0] = 1.  # Water component
    if M == 2:
        # F = 9A+(6(CL-4)+UD(2P2U-8))B+6C+4UD(1-P2U)D+6E+2UDP2UF+2G+2H+I+2UDJ
        alpha[1, 1:] = [9, 6*(CL-4)+UD*(2*P2U-8), 6, 4*UD*(1-P2U), 6, 2*UD*P2U,
                        2, 2, 1, UD*2]
    elif M == 3:
        # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # F2 = (2P2U-8)B+4(1-P2U)D+2P2UF+2J
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, 2*P2U-8, 0, 4*(1-P2U), 0, 2*P2U, 0, 0, 0, 2]
    elif M == 4:
        # F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # F2 = -8B+4D+2J
        # F3 = 2B-4D+2F
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
    elif M == 5:
        # F1 = 9A-24B+6C+6E+2G+2H+I
        # F2 = -8B+4D+2J
        # F3 = 2B-4D+2F
        # F4 = 6B
        alpha[1, 1:] = [9, -24, 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
        alpha[4, 1:] = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]
    return alpha


# Update model parameter object mPar and set default parameters
def setupModelParams(mPar, clockwisePrecession=False, temperature=None):
    if 'watcs' in mPar:
        watCS = [float(mPar.watcs)]
    else:
        if temperature: # Temperature dependence according to Hernando 2014
            watCS = 1.3 + 3.748 -.01085 * temperature # Temp in [°C]
        else:
            watCS = [4.7]
    if 'fatcs' in mPar:
        fatCS = [float(cs) for cs in mPar.fatcs.split(',')]
    else:
        fatCS = [1.3]
    mPar.CS = np.array(watCS+fatCS, dtype=np.float32)
    if clockwisePrecession:
        mPar.CS *= -1
    mPar.P = len(mPar.CS)
    if 'nfac' in mPar:
        mPar.nFAC = int(mPar.nfac)
    else:
        mPar.nFAC = 0
    if mPar.nFAC > 0 and mPar.P is not 11:
        raise Exception(
            'FAC excpects exactly one water and ten triglyceride resonances')
    mPar.M = 2+mPar.nFAC
    if 'cl' in mPar:
        mPar.CL = float(mPar.cl)
    else:
        mPar.CL = 17.4  # Derived from Lundbom 2010
    if 'p2u' in mPar:
        mPar.P2U = float(mPar.p2u)
    else:
        mPar.P2U = 0.2  # Derived from Lundbom 2010
    if 'ud' in mPar:
        mPar.UD = float(mPar.ud)
    else:
        mPar.UD = 2.6  # Derived from Lundbom 2010
    if mPar.nFAC == 0:
        mPar.alpha = np.zeros([mPar.M, mPar.P], dtype=np.float32)
        mPar.alpha[0, 0] = 1.
        if 'relamps' in mPar:
            for (p, a) in enumerate(mPar.relamps.split(',')):
                mPar.alpha[1, p+1] = float(a)
        else:
            for p in range(1, mPar.P):
                mPar.alpha[1, p] = float(1/len(fatCS))
    elif mPar.nFAC == 1:
        mPar.alpha = getFACalphas(mPar.CL, mPar.P2U)
    elif mPar.nFAC == 2:
        mPar.alpha = getFACalphas(mPar.CL)
    elif mPar.nFAC == 3:
        mPar.alpha = getFACalphas()
    else:
        raise Exception('Unknown number of FAC parameters: {}'
                        .format(mPar.nFAC))

    # For Fatty Acid Composition, create modelParams for two passes: mPar and mPar.pass2
    # First pass: use standard fat-water separation to determine B0 and R2*
    # Second pass: do the Fatty Acid Composition
    if mPar.nFAC > 0: 
        mPar.pass2 = AttrDict(mPar) # copy mPar into pass 2, then modify pass 1
        mPar.alpha = getFACalphas(mPar.CL, mPar.P2U, mPar.UD)
        mPar.M = mPar.alpha.shape[0]


# group slices in sliceList in slabs of reconSlab contiguous slices
def getSlabs(sliceList, reconSlab):
    slabs = []
    slices = []
    pos = 0
    for z, slice in enumerate(sliceList):
        # start a new slab
        if slices and (len(slices) == reconSlab or not slice == slices[-1]+1):
            slabs.append((slices, pos))
            slices = [slice]
            pos = z
        else:
            slices.append(slice)
    slabs.append((slices, pos))
    return slabs


# Convert string on form "0-3, 5, 8-22" to set of integers
def readIntString(str):
    ints = []
    for word in [w.replace(' ', '') for w in str.split(',')]:
        if word.isdigit():
            ints.append(int(word))
        elif '-' in word:
            try:
                digits = [int(d.replace(' ', '')) for d in word.split('-')]
            except ValueError:
                raise Exception('Unexpected integer string "{}"'.format(str))
            if len(digits) > 2 or digits[0] > digits[1]:
                raise Exception('Unexpected integer string "{}"'.format(str))
            else:
                for i in range(digits[0], digits[1]+1):
                    ints.append(i)
    return list(set(ints))


# Get list of all files in directories in dirList
def getFiles(dirList):
    files = []  # Get list of files:
    for dir in dirList:
        files = files+[os.path.join(dir, file) for file in os.listdir(dir)]
    return files

    
# Update data param object, set default parameters and read data from files
def setupDataParams(dPar, outDir=None):
    if outDir:
        dPar.outDir = outDir
    elif 'outdir' in dPar:
        dPar.outDir = dPar.outdir
    else:
        raise Exception('No outDir defined')
    # Rescaling might be needed for datasets with too small/large pixel values
    if 'rescale' in dPar:
        dPar.reScale = float(dPar.rescale)
    else:
        dPar.reScale = 1.0
    if 'echoes' in dPar:
        dPar.echoes = readIntString(dPar.echoes)
    if 'slicelist' in dPar:
        dPar.sliceList = readIntString(dPar.slicelist)
    if 'cropfov' in dPar:
        dPar.cropFOV = [int(x.replace(' ', '')) for x in dPar.cropfov.split(',')]
    if 'reconslab' in dPar:
        dPar.reconSlab = int(dPar.reconslab)
    if 'temperature' in dPar:
        dPar.Temperature = float(dPar.temperature)
    else:
        dPar.Temperature = None
    if 'clockwiseprecession' in dPar:
        dPar.clockwisePrecession = dPar.clockwiseprecession == 'True'
    else:
        dPar.clockwisePrecession = False
    if 'offrescenter' in dPar:
        dPar.offresCenter = float(dPar.offrescenter)
    else:
        dPar.offresCenter = 0.
    if 'files' in dPar:
        dPar.files = dPar.files.split(',')
    else:
        dPar.files = []
    if 'dirs' in dPar:
        dPar.dirs = dPar.dirs.split(',')
        dPar.files += getFiles(dPar.dirs)
    validFiles = DICOM.getValidFiles(dPar['files'])
    if validFiles:
        DICOM.updateDataParams(dPar, validFiles)
    else:
        if len(dPar.files) == 1 and dPar.files[0][-4:] == '.mat':
            MATLAB.updateDataParams(dPar, dPar.files[0])
        else:
            raise Exception('No valid files found')
    if 'reconSlab' in dPar:
        dPar.slabs = getSlabs(dPar.sliceList, dPar.reconSlab)


# Read configuration file
def readConfig(file, section):
    config = configparser.ConfigParser()
    config.read(file)
    return AttrDict(config[section])