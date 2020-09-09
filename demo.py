#!/usr/bin/env python3

import main
import numpy as np
import pydicom
import scipy.io
import os
import time


def getScore(case, dir):
    # Read reconstructed MATLAB-file
    file = os.path.join(dir, '0.mat')
    try:
        mat = scipy.io.loadmat(file)
    except:
        raise Exception('Could not read MATLAB file {}'.format(file))
    recFF = mat['ff'].flatten(order='F')/100
    recFF.shape = recFF.shape + (1,)
    # Read reference MATLAB-file
    refFile = r'./challenge/refdata.mat'
    try:
        mat = scipy.io.loadmat(refFile)
    except:
        raise Exception('Could not read MATLAB file {}'.format(refFile))
    refFF = mat['REFCASES'][0, case - 1]
    mask = mat['MASKS'][0, case - 1]
    # Calculate score
    score = 100 * \
        (1 - np.sum((np.abs(refFF - recFF) > 0.1) * mask) / np.sum(mask))
    return score


if __name__ == '__main__':

    if not os.path.isfile(r'./challenge/refdata.mat'):
        url = r'"https://dl.dropboxusercontent.com/u/5131732/' + \
            r'ISMRM_Challenge/refdata.mat"'
        raise Exception(r'Please download ISMRM 2012 challenge reference data '
                        'from {} and put in "challenge" subdirectory'.format(url))

    modelParamsFile = r'./modelParams.txt'
    cases = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    for case in cases:
        if not os.path.isfile(r'./challenge/{}.mat'.format(str(case).zfill(2))):
            url = r'"http://dl.dropbox.com/u/5131732/' + \
                r'ISMRM_Challenge/data_matlab.zip"'
            raise Exception(
                r'Please download ISMRM 2012 challenge datasets from {} '
                r'and put in "challenge" subdirectory'.format(url))

    results = []
    for case in cases:
        dataParamsFile = r'./challenge/{}.txt'.format(case)
        if case == 9:
            algoParamsFile = r'./algoParams2D.txt'
        else:
            algoParamsFile = r'./algoParams3D.txt'
        outDir = r'./challenge/{}_REC'.format(str(case).zfill(2))
        t = time.time()
        main.main(dataParamsFile, algoParamsFile, modelParamsFile, outDir)
        results.append((case, getScore(case, outDir), time.time() - t))

    print()
    for case, score, recTime in results:
        print('Case {}: score {}% in {} sec'.format(
            case, round(score, 2), round(recTime, 1)))
