import FWQPBO
import numpy as np
import dicom
import scipy.io
import os
import time

def getScore(case,dir):
    # Read reconstructed DICOM-file
    recFF = np.array([])
    for file in os.listdir(dir):
        try: dcm = dicom.read_file(os.path.join(dir,file))
        except: raise Exception('File not found: {}'.format(file))
        reScaleSlope = dcm[0x00281053].value
        recFF = np.concatenate((recFF,dcm.pixel_array.transpose().flatten()*reScaleSlope))
    recFF.shape=recFF.shape+(1,)
    # Read reference MATLAB-file
    refFile = r'.\challenge\refdata.mat'
    try: mat = scipy.io.loadmat(refFile)
    except: raise Exception('Could not read MATLAB file {}'.format(refFile))
    refFF = mat['REFCASES'][0,case-1]
    mask = mat['MASKS'][0,case-1]    
    # Calculate score
    score = 100*(1-np.sum((np.abs(refFF-recFF)>0.1)*mask)/np.sum(mask))
    return score

if not os.path.isfile(r'.\challenge\refdata.mat'):
    raise Exception(r'Please download ISMRM 2012 challenge reference data from "https://dl.dropboxusercontent.com/u/5131732/ISMRM_Challenge/refdata.mat" and put in "challenge" subdirectory')
    
modelParamsFile = r'.\modelParams.txt'
cases = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

for case in cases:
    if not os.path.isfile(r'.\challenge\{}.mat'.format(str(case).zfill(2))):
        raise Exception(r'Please download ISMRM 2012 challenge datasets from "http://dl.dropbox.com/u/5131732/ISMRM_Challenge/data_matlab.zip" and put in "challenge" subdirectory')

results = []
for case in cases:
    dataParamsFile = r'.\challenge\{}.txt'.format(case)
    if case==9: algoParamsFile = r'.\algoParams2D.txt'
    else: algoParamsFile = r'.\algoParams3D.txt'
    outDir = r'.\challenge\{}_REC'.format(str(case).zfill(2))
    t = time.time()
    FW.FW(dataParamsFile,algoParamsFile,modelParamsFile,outDir)
    results.append((case,getScore(case,outDir+'/FF'),time.time()-t))

print()
for case,score,time in results: print('Case {}: score {}% in {} sec'.format(case,round(score,2),round(time,1)))