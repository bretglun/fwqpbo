#!/usr/bin/env python3

import numpy as np
import os
import pydicom
import datetime
import sys
import configparser
import optparse
import scipy.io
import fatWaterSeparation


# Helper class for convenient reading of config files
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


gyro = 42.58  # 1H gyromagnetic ratio


# Dictionary of DICOM tags
tagDict = {
    'Image Type': 0x00080008,
    'SOP Class UID': 0x00080016,
    'SOP Instance UID': 0x00080018,
    'Series Description': 0x0008103E,
    'Slice Thickness': 0x00180050,
    'Spacing Between Slices': 0x00180088,
    'Echo Time': 0x00180081,
    'Imaging Frequency': 0x00180084,
    'Protocol Name': 0x00181030,
    'Study Instance UID': 0x0020000D,
    'Series Instance UID': 0x0020000E,
    'Series Number': 0x00200011,
    'Slice Location': 0x00201041,
    'Image Position (Patient)': 0x00200032,
    'Rows': 0x00280010,
    'Columns': 0x00280011,
    'Pixel Spacing': 0x00280030,
    'Pixel Aspect Ratio': 0x00280034,
    'Smallest Pixel Value': 0x00280106,
    'Largest Pixel Value': 0x00280107,
    'Window Center': 0x00281050,
    'Window Width': 0x00281051,
    'Rescale Intercept': 0x00281052,
    'Rescale Slope': 0x00281053,
    'Number of frames': 0x00280008,
    'Frame sequence': 0x52009230}  # Per-frame Functional Groups Sequence


def getSOPInstanceUID():
    t = datetime.datetime.now()
    datestr = '{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}{:03d}'.format(
     t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond//1000)
    randstr = str(np.random.randint(1000, 1000000000))
    uidstr = "1.3.12.2.1107.5.2.32.35356." + datestr + randstr
    return uidstr


def getSeriesInstanceUID(dPar, seriesDescription):
    if not 'seriesInstanceUIDs' in dPar:
        dPar['seriesInstanceUIDs'] = {}
    if not seriesDescription in dPar['seriesInstanceUIDs']:
        dPar['seriesInstanceUIDs'][seriesDescription] = getSOPInstanceUID() + ".0.0.0"
    return dPar['seriesInstanceUIDs'][seriesDescription]


# Set window so that 95% of pixels are inside
def get95percentileWindow(im, intercept, slope):
    lims = np.percentile(im, [2.5, 97.5])
    width = lims[1]-lims[0]
    center = width/2.+lims[0]
    return center*slope+intercept, width*slope


# Sets DICOM element value in dataset ds at tag=key. Use frame for multiframe
# DICOM files. If missing tag, a new one is created if value representation VR
# is provided
def setTagValue(ds, key, val, frame=None, VR=None):
    # TODO: improve support for multi-frame DICOM images
    # Philips(?) private tag containing frame tags
    if (frame is not None and
       0x2005140f in ds[tagDict['Frame sequence']].value[frame]):
        frameObject = ds[tagDict['Frame sequence']].value[frame][0x2005140f][0]
        if tagDict[key] in frameObject:
            frameObject[tagDict[key]].value = val
            return True
    if tagDict[key] in ds:
        ds[tagDict[key]].value = val
        return True
    # Else, add as new DICOM element:
    if VR:
        # Philips(?) private tag containing frame tags
        if (frame is not None and
           0x2005140f in ds[tagDict['Frame sequence']].value[frame]):
            frameObject = ds[tagDict['Frame sequence']].\
                value[frame][0x2005140f][0]
            frameObject.add_new(tagDict[key], VR, val)
            return True
        ds.add_new(tagDict[key], VR, val)
        return True
    # print('Warning: DICOM tag {} was not set'.format(key))
    return False


imgTypes = {
    'phi': {'descr': 'Initial phase (degrees)', 'seriesNumber': 100},
    'wat': {'descr': 'Water-only', 'seriesNumber': 101},
    'fat': {'descr': 'Fat-only', 'seriesNumber': 102},
    'ip': {'descr': 'In-phase', 'seriesNumber': 103},
    'op': {'descr': 'Opposed-phase', 'seriesNumber': 104},
    'ff': {'descr': 'Fat Fraction (%)', 'seriesNumber': 105, 'reScaleIntercept': -100, 'reScaleSlope': 0.1},
    'R2map': {'descr': 'R2* (msec-1)', 'seriesNumber': 106},
    'B0map': {'descr': 'B0 inhomogeneity (ppm)', 'seriesNumber': 107, 'reScaleSlope': 0.001},
    'CL': {'descr': 'FAC Chain length', 'seriesNumber': 108, 'reScaleSlope': 0.01},
    'UD': {'descr': 'FAC Unsaturation degree', 'seriesNumber': 109, 'reScaleSlope': 0.01},
    'PUD': {'descr': 'FAC Polyunsaturation degree', 'seriesNumber': 110, 'reScaleSlope': 0.01}
}


# Save numpy array to DICOM image.
# Based on input DICOM image if exists, else create from scratch
def saveDICOMseries(outDir, imgType, img, dPar):
    seriesDescription = imgTypes[imgType]['descr']
    seriesNumber = imgTypes[imgType]['seriesNumber']
    if 'reScaleIntercept' in imgTypes[imgType]:
        reScaleIntercept = imgTypes[imgType]['reScaleIntercept']
    else:
        reScaleIntercept = 0.
    if 'reScaleSlope' in imgTypes[imgType]:
        reScaleSlope = imgTypes[imgType]['reScaleSlope']
    else:
        reScaleSlope = 1.
    
    seriesInstanceUID = getSeriesInstanceUID(dPar, seriesDescription)
    # Single file is interpreted as multi-frame
    multiframe = dPar.frameList and \
        len(set([frame[0] for frame in dPar.frameList])) == 1
    if multiframe:
        ds = pydicom.read_file(dPar.frameList[0][0])
        imVol = np.empty([dPar.nz, dPar.ny*dPar.nx], dtype='uint16')
        frames = []
    if dPar.frameList:
        DICOMimgType = getType(dPar.frameList)
    for z, slice in enumerate(dPar.sliceList):
        filename = outDir+r'/{}.dcm'.format(slice)
        # Extract slice, scale and type cast pixel data
        pixelData = np.array([max(0, (val-reScaleIntercept)/reScaleSlope)
                      for val in img[:, :, z].flatten()])
        pixelData = pixelData.astype('uint16')
        # Set window so that 95% of pixels are inside
        windowCenter, windowWidth = get95percentileWindow(
                                            pixelData, reScaleIntercept, reScaleSlope)
        if dPar.frameList:
            # Get frame
            frame = dPar.frameList[dPar.totalN*slice*len(DICOMimgType)]
            iFrame = frame[1]
            if not multiframe:
                ds = pydicom.read_file(frame[0])
        else:
            iFrame = None
            # Create new DICOM images from scratch
            file_meta = pydicom.dataset.Dataset()
            file_meta.MediaStorageSOPClassUID = \
                'Secondary Capture Image Storage'
            file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.' +\
                '1.1.111165684411017669021768385720736873780'
            file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.' + \
                '1.0.100.4.0'
            file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            ds = pydicom.dataset.FileDataset(filename, {}, file_meta=file_meta,
                                           preamble=b"\0"*128)
            # Add DICOM tags:
            ds.Modality = 'WSD'
            ds.ContentDate = str(datetime.date.today()).replace('-', '')
            ds.ContentTime = str(datetime.time())  # millisecs since the epoch
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelRepresentation = 0
            ds.HighBit = 15
            ds.BitsStored = 16
            ds.BitsAllocated = 16
            ds.SmallestImagePixelValue = '\\x00\\x00'
            ds.LargestImagePixelValue = '\\xff\\xff'
            ds.Columns = img.shape[1]
            ds.Rows = img.shape[0]
            setTagValue(ds, 'Study Instance UID',
                        getSOPInstanceUID(), iFrame, 'UI')
            setTagValue(ds, 'Pixel Spacing', [dPar.dx, dPar.dy], iFrame, 'DS')
            setTagValue(ds, 'Pixel Aspect Ratio', [int(dPar.dx*100), int(dPar.dy*100)], iFrame, 'IS')
        # Change/add DICOM tags:
        setTagValue(ds, 'SOP Instance UID', getSOPInstanceUID(), iFrame, 'UI')
        setTagValue(ds, 'Series Instance UID', seriesInstanceUID, iFrame, 'UI')
        setTagValue(ds, 'Series Number', seriesNumber, iFrame, 'IS')
        setTagValue(ds, 'Echo Time', 0., iFrame, 'DS')
        setTagValue(ds, 'Protocol Name', 'Derived Image', iFrame, 'LO')
        setTagValue(ds, 'Series Description', seriesDescription, iFrame, 'LO')
        setTagValue(ds, 'Smallest Pixel Value', np.min(pixelData), iFrame)
        setTagValue(ds, 'Largest Pixel Value', np.max(pixelData), iFrame)
        setTagValue(ds, 'Window Center', int(windowCenter), iFrame, 'DS')
        setTagValue(ds, 'Window Width', int(windowWidth), iFrame, 'DS')
        setTagValue(ds, 'Rescale Intercept', reScaleIntercept, iFrame, 'DS')
        setTagValue(ds, 'Rescale Slope', reScaleSlope, iFrame, 'DS')

        if multiframe:
            imVol[z] = pixelData
            frames.append(iFrame)
        else:
            ds.PixelData = pixelData
            ds.save_as(filename)

    if multiframe:
        setTagValue(ds, 'SOP Instance UID', getSOPInstanceUID())
        setTagValue(ds, 'Number of frames', len(frames))
        ds[tagDict['Frame sequence']].value = \
            [ds[tagDict['Frame sequence']].value[frame] for frame in frames]
        ds.PixelData = imVol
        filename = outDir+r'/0.dcm'
        ds.save_as(filename)


# Zero pad back any cropped FOV
def padCropped(croppedImage, dPar):
    if 'cropFOV' in dPar:
        image = np.zeros((dPar.nz, dPar.Ny, dPar.Nx))
        x1, x2 = dPar.cropFOV[0], dPar.cropFOV[1]
        y1, y2 = dPar.cropFOV[2], dPar.cropFOV[3]
        image[:, y1:y2, x1:x2] = croppedImage
        return image
    else:
        return croppedImage


# Save numpy array as MatLab array
def saveMatLab(output, dPar):
    filename = dPar.outDir+r'/{}.mat'.format(dPar.sliceList[0])
    scipy.io.savemat(filename, output)


# Save all data in output as DICOM images
def saveDICOM(output, dPar):
    for seriesType in output:
        outDir = os.path.join(dPar.outDir, seriesType)
        if not os.path.isdir(outDir):
            os.mkdir(outDir)
        print(r'Writing image{} to "{}"'.format('s'*(dPar.nz > 1), outDir))
        saveDICOMseries(outDir, seriesType, output[seriesType], dPar)


def save(output, dPar):
    if not os.path.isdir(dPar.outDir):
        os.mkdir(dPar.outDir)
    
    for seriesType in output: # zero pad if was cropped and reshape to row,col,slice
        output[seriesType] = np.moveaxis(padCropped(output[seriesType].reshape((dPar.nz, dPar.ny, dPar.nx)), dPar), 0, -1)
    
    if dPar.fileType == 'DICOM':
        saveDICOM(output, dPar)
    elif dPar.fileType == 'MatLab':
        saveMatLab(output, dPar)
    else:
        raise Exception('Unknown filetype: {}'.format(dPar.fileType))


# Merge output for slices reconstructed separately
def mergeOutputSlices(outputList):
    mergedOutput = outputList[0]
    for output in outputList[1:]:
        for seriesType in output:
            mergedOutput[seriesType] = np.concatenate((mergedOutput[seriesType], output[seriesType]))
    return mergedOutput


# Check if ds is a multiframe DICOM object
def isMultiFrame(ds):
    return tagDict['Number of frames'] in ds and \
        int(ds[tagDict['Number of frames']].value) > 1 and \
        tagDict['Frame sequence'] in ds


# Get list of all files in directories in dirList
def getFiles(dirList):
    files = []  # Get list of files:
    for dir in dirList:
        files = files+[os.path.join(dir, file) for file in os.listdir(dir)]
    return files


# Retrieves DICOM element value from dataset ds at tag=key.
# Use frame for multiframe DICOM files
def getTagValue(ds, key, frame=None):
    # Philips(?) private tag containing frame tags
    if (frame is not None and
       0x2005140f in ds[tagDict['Frame sequence']].value[frame]):
        frameObject = ds[tagDict['Frame sequence']].value[frame][0x2005140f][0]
        if tagDict[key] in frameObject:
            return frameObject[tagDict[key]].value
    if tagDict[key] in ds:
        return ds[tagDict[key]].value
    return None


# Translates series description tag to M/P/R/I for magn/phase/real/imaginary
def seriesDescription2type(seriesDescription):
    for type, description in [('R', 'Real Image'), ('I', 'Imag Image')]:
        if description in seriesDescription:
            return type
    return None


# Translates image type tag to M/P/R/I for magnitude/phase/real/imaginary
def typeTag2type(tagValue):
    for type in ['M', 'P', 'R', 'I']:
        if type in tagValue:
            return type
    return None


# Retrieve attribute from dataset ds. Use frame for multiframe DICOM files.
# Must attributes are read directly from their corresponding DICOM tag
def getAttribute(ds, attr, frame=None):
    attribute = getTagValue(ds, attr, frame)
    if attr == 'Slice Location' and attribute is None:
        attribute = getTagValue(ds, 'Image Position (Patient)', frame)
        if attribute:
            attribute = attribute[2]
    elif attr == 'Spacing Between Slices' and attribute is None:
        attribute = getTagValue(ds, 'Slice Thickness', frame)
    elif attr == 'Image Type' and attribute:  # special handling of image type
        attribute = typeTag2type(attribute)
        if not attribute:
            attribute = seriesDescription2type(
                getTagValue(ds, 'Series Description', frame))
    return attribute


# Check if attribute is in DICOM dataset ds
def AttrInDataset(ds, attr, multiframe):
    if getAttribute(ds, attr) is not None:
        return True
    elif multiframe:
        for frame in range(len(ds[tagDict['Frame sequence']].value)):
            if not getAttribute(ds, attr, frame):
                return False  # Attribute must be in all frames!
        return True
    return False

# List of DICOM attributes required for the water-fat separation
reqAttributes = ['Image Type', 'Echo Time', 'Slice Location',
                               'Imaging Frequency', 'Columns', 'Rows',
                               'Pixel Spacing', 'Spacing Between Slices']


# Checks if list of DICOM files contains required information
def isValidDataset(files, printOutput=False):
    frameList = []
    for file in files:
        ds = pydicom.read_file(file, stop_before_pixels=True)
        multiframe = isMultiFrame(ds)
        if multiframe:  # Multi-frame DICOM files
            if len(files) > 1:
                raise Exception('Support for multiple multi-frame DICOM ' +
                                'files not implemented yet!')
            for frame in range(len(ds[tagDict['Frame sequence']].value)):
                frameList.append([file]+[frame]+[getAttribute(ds, attr, frame)
                                 for attr in reqAttributes])
        else:  # Single-frame DICOM files
            frameList.append([file]+[None]+[getAttribute(ds, attr)
                             for attr in reqAttributes])
    try:
        getType(frameList)
    except Exception as e:
        if printOutput:
            print(e)
        return False
    if len(set([tags[3] for tags in frameList])) < 3:
        if printOutput:
            print('Error: Less than three echo times in dataset')
        return False
    if len(set([tags[5] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple imaging frequencies in dataset')
        return False
    if len(set([tags[6] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple image sizes (y-dir) in dataset')
        return False
    if len(set([tags[7] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple image sizes (x-dir) in dataset')
        return False
    if len(set([tags[8][0] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple voxel sizes (y-dir) in dataset')
        return False
    if len(set([tags[8][1] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple voxel sizes (x-dir) in dataset')
        return False
    if len(set([tags[9] for tags in frameList])) > 1:
        if printOutput:
            print('Error: Multiple slice thicknesses in dataset')
        return False
    return True


# Extract files that are readable and have all required DICOM tags
def getValidFiles(files, printOutput=False):
    validFiles = []
    for file in files:
        try:
            ds = pydicom.read_file(file, stop_before_pixels=True)
        except:
            if printOutput:
                print('Could not read file: {}'.format(file))
            continue
        multiframe = isMultiFrame(ds)
        hasRequiredAttrs = [AttrInDataset(ds, attr, multiframe)
                            for attr in reqAttributes]
        if not all(hasRequiredAttrs):
            if printOutput:
                print('File {} is missing required DICOM tags:'.format(file))
                for i, hasAttr in enumerate(hasRequiredAttrs):
                    if not hasAttr:
                        print(reqAttributes[i])
            continue
        else:
            validFiles.append(file)
    return validFiles


# get combination of image types for DICOM frames in frameList
def getType(frameList, printType=False):
    typeTags = [tags[2] for tags in frameList]
    numR = typeTags.count('R')
    numI = typeTags.count('I')
    numM = typeTags.count('M')
    numP = typeTags.count('P')
    if numM+numP == 0 and numR+numI > 0 and numR == numI:
        if printType:
            print('Real/Imaginary images')
        return 'RI'
    elif numM+numP > 0 and numR+numI == 0 and numM == numP:
        if printType:
            print('Magnitude/Phase images')
        return 'MP'
    elif numP == 0 and numM+numR+numI > 0 and numM == numR == numI:
        if printType:
            print('Magnitude/Real/Imaginary images')
        return 'MRI'
    elif numM+numP+numR+numI > 0 and numM == numP == numR == numI:
        if printType:
            print('Magnitude/Phase/Real/Imaginary images')
        return 'MPRI'
    else:
        raise Exception('Unknown combination of image types: ' +
                        '{} real, {} imag, {} magn, {} phase'
                        .format(numR, numI, numM, numP))


# update dPar with info retrieved from the DICOM files including image data
def updateDataParamsDICOM(dPar, files):
    dPar.fileType = 'DICOM'
    frameList = []
    for file in files:
        ds = pydicom.read_file(file, stop_before_pixels=True)
        multiframe = isMultiFrame(ds)
        if multiframe:
            if len(files) > 1:
                raise Exception('Support for multiple multi-frame DICOM' +
                                'files not implemented yet!')
            for frame in range(len(ds[tagDict['Frame sequence']].value)):
                frameList.append([file]+[frame]+[getAttribute(ds, attr, frame)
                                 for attr in reqAttributes])
        else:  # Single frame DICOM files
            frameList.append([file]+[None]+[getAttribute(ds, attr)
                             for attr in reqAttributes])
    frameList.sort(key=lambda tags: tags[2])  # First, sort on type (M/P/R/I)
    frameList.sort(key=lambda tags: tags[3])  # Second, sort on echo time
    frameList.sort(key=lambda tags: tags[4])  # Third, sort on slice location

    type = getType(frameList, True)
    dPar.dx = float(frameList[0][8][1])
    dPar.dy = float(frameList[0][8][0])
    dPar.dz = float(frameList[0][9])

    dPar.B0 = frameList[0][5]/gyro
    # [msec]->[sec]
    echoTimes = sorted(set([float(tags[3])/1000. for tags in frameList]))
    dPar.totalN = len(echoTimes)
    if 'echoes' not in dPar:
        dPar.echoes = range(dPar.totalN)
    echoTimes = [echoTimes[echo] for echo in dPar.echoes]
    dPar.N = len(dPar.echoes)
    if dPar.N < 2:
        raise Exception('At least 2 echoes required, only {} given'
                        .format(dPar.N))
    dPar.t1 = echoTimes[0]
    dPar.dt = np.mean(np.diff(echoTimes))
    if np.max(np.diff(echoTimes))/dPar.dt > 1.05 or \
       np.min(np.diff(echoTimes))/dPar.dt < .95:
        print('Warning: echo inter-spacing varies more than 5%')
        print(echoTimes)
    nSlices = len(set([tags[4] for tags in frameList]))
    if 'sliceList' not in dPar:
        dPar.sliceList = range(nSlices)

    dPar.nx = frameList[0][6]
    dPar.ny = frameList[0][7]
    dPar.nz = len(dPar.sliceList)

    if 'cropFOV' in dPar:
        x1, x2 = dPar.cropFOV[0], dPar.cropFOV[1]
        y1, y2 = dPar.cropFOV[2], dPar.cropFOV[3]
        dPar.Nx, dPar.nx = dPar.nx, x2-x1
        dPar.Ny, dPar.ny = dPar.ny, y2-y1
    else:
        x1, x2 = 0, dPar.nx
        y1, y2 = 0, dPar.ny
    img = []
    if multiframe:
        file = frameList[0][0]
        dcm = pydicom.read_file(file)
    for n in dPar.echoes:
        for slice in dPar.sliceList:
            i = (dPar.N*slice+n)*len(type)
            if type == 'MP':  # Magnitude/phase images
                magnFrame = i
                phaseFrame = i+1
                if multiframe:
                    magn = dcm.pixel_array[
                        frameList[magnFrame][1]][y1:y2, x1:x2].flatten()
                    phase = dcm.pixel_array[
                        frameList[phaseFrame][1]][y1:y2, x1:x2].flatten()
                    # Abs val needed for Siemens data to get correct phase sign
                    reScaleIntercept = \
                        np.abs(getAttribute(dcm, 'Rescale Intercept',
                                            frameList[phaseFrame][1]))
                else:
                    magnFile = frameList[magnFrame][0]
                    phaseFile = frameList[phaseFrame][0]
                    mDcm = pydicom.read_file(magnFile)
                    pDcm = pydicom.read_file(phaseFile)
                    magn = mDcm.pixel_array[y1:y2, x1:x2].flatten()
                    phase = pDcm.pixel_array[y1:y2, x1:x2].flatten()
                    # Abs val needed for Siemens data to get correct phase sign
                    try:
                        reScaleIntercept = np.abs(
                            getAttribute(pDcm, 'Rescale Intercept'))
                    except:
                        print('No Rescale Intercept DICOM tag found. Using 4096.')
                        reScaleIntercept = 4096
                # For some reason, intercept is used as slope (Siemens only?)
                c = magn*np.exp(phase/float(reScaleIntercept)*2*np.pi*1j)
            # Real/imaginary images and Magnitude/real/imaginary images
            elif type in ['RI', 'MRI', 'MPRI']:
                if type == 'RI':
                    realFrame = i+1
                elif type == 'MRI':
                    realFrame = i+2
                elif type == 'MPRI':
                    realFrame = i+3
                imagFrame = i
                if multiframe:
                    realPart = dcm.pixel_array[
                        frameList[realFrame][1]][y1:y2, x1:x2].flatten()
                    imagPart = dcm.pixel_array[
                        frameList[imagFrame][1]][y1:y2, x1:x2].flatten()
                    # Assumes real and imaginary slope/intercept are equal
                    reScaleIntercept = getAttribute(
                        dcm, 'Rescale Intercept', frameList[realFrame][1])
                    reScaleSlope = getAttribute(
                        dcm, 'Rescale Slope', frameList[realFrame][1])
                else:
                    realFile = frameList[realFrame][0]
                    imagFile = frameList[imagFrame][0]
                    rDcm = pydicom.read_file(realFile)
                    iDcm = pydicom.read_file(imagFile)
                    realPart = rDcm.pixel_array[y1:y2, x1:x2].flatten()
                    imagPart = iDcm.pixel_array[y1:y2, x1:x2].flatten()
                    # Assumes real and imaginary slope/intercept are equal
                    reScaleIntercept = getAttribute(rDcm, 'Rescale Intercept')
                    reScaleSlope = getAttribute(rDcm, 'Rescale Slope')
                if reScaleIntercept and reScaleSlope:
                    offset = reScaleIntercept/reScaleSlope
                else:
                    offset = -2047.5
                c = (realPart+offset)+1.0*1j*(imagPart+offset)
            else:
                raise Exception('Unknown image types')
            img.append(c)
    dPar.frameList = frameList
    dPar.img = np.array(img)*dPar.reScale


# update dPar with information retrieved from MATLAB file arranged
# according to ISMRM fat-water toolbox
def updateDataParamsMATLAB(dPar, file):
    dPar.fileType = 'MatLab'
    try:
        mat = scipy.io.loadmat(file)
    except:
        raise Exception('Could not read MATLAB file {}'.format(file))
    data = mat['imDataParams'][0, 0]

    for i in range(0, 4):
        if len(data[i].shape) == 5:
            img = data[i]  # Image data (row,col,slice,coil,echo)
        elif data[i].shape[1] > 2:
            echoTimes = data[i][0]  # TEs [sec]
        else:
            if data[i][0, 0] > 1:
                dPar.B0 = data[i][0, 0]  # Fieldstrength [T]
            else:
                clockwise = data[i][0, 0]  # Clockwiseprecession?

    if clockwise != 1:
        raise Exception('Warning: Not clockwise precession. ' +
                        'Need to write code to handle this case!')

    dPar.ny, dPar.nx, dPar.nz, nCoils, dPar.N = img.shape
    if nCoils > 1:
        raise Exception('Warning: more than one coil. ' +
                        'Need to write code to coil combine!')

    # Get only slices in dPar.sliceList
    if 'sliceList' not in dPar:
        dPar.sliceList = range(dPar.nz)
    else:
        img = img[:, :, dPar.sliceList, :, :]
        dPar.nz = len(dPar.sliceList)
    # Get only echoes in dPar.echoes
    dPar.totalN = dPar.N
    if not hasattr(dPar, 'echoes'):
        dPar.echoes = range(dPar.totalN)
    else:
        img = img[:, :, :, :, dPar.echoes]
        echoTimes = echoTimes[dPar.echoes]
        dPar.N = len(dPar.echoes)
    if 'cropFOV' in dPar:
        x1, x2 = dPar.cropFOV[0], dPar.cropFOV[1]
        y1, y2 = dPar.cropFOV[2], dPar.cropFOV[3]
        dPar.Nx, dPar.nx = dPar.nx, x2-x1
        dPar.Ny, dPar.ny = dPar.ny, y2-y1
        img = img[y1:y2, x1:x2, :]
    if dPar.N < 2:
        raise Exception(
            'At least 2 echoes required, only {} given'.format(dPar.N))
    dPar.t1 = echoTimes[0]
    dPar.dt = np.mean(np.diff(echoTimes))
    if np.max(np.diff(echoTimes))/dPar.dt > 1.05 or np.min(
      np.diff(echoTimes))/dPar.dt < .95:
        raise Exception('Warning: echo inter-spacing varies more than 5%')

    dPar.frameList = []

    dPar.dx, dPar.dy, dPar.dz = 1.5, 1.5, 5  # Ad hoc assumption on voxelsize

    # To get data as: (echo,slice,row,col)
    img.shape = (dPar.ny, dPar.nx, dPar.nz, dPar.N)
    img = np.transpose(img)
    img = np.swapaxes(img, 2, 3)

    img = img.flatten()
    dPar.img = img*dPar.reScale


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
        # // F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # // F2 = (2P2U-8)B+4(1-P2U)D+2P2UF+2J
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, 2*P2U-8, 0, 4*(1-P2U), 0, 2*P2U, 0, 0, 0, 2]
    elif M == 4:
        # // F1 = 9A+6(CL-4)B+6C+6E+2G+2H+I
        # // F2 = -8B+4D+2J
        # // F3 = 2B-4D+2F
        alpha[1, 1:] = [9, 6*(CL-4), 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
    elif M == 5:
        # // F1 = 9A-24B+6C+6E+2G+2H+I
        # // F2 = -8B+4D+2J
        # // F3 = 2B-4D+2F
        # // F4 = 6B
        alpha[1, 1:] = [9, -24, 6, 0, 6, 0, 2, 2, 1, 0]
        alpha[2, 1:] = [0, -8, 0, 4, 0, 0, 0, 0, 0, 2]
        alpha[3, 1:] = [0, 2, 0, -4, 0, 2, 0, 0, 0, 0]
        alpha[4, 1:] = [0, 6, 0, 0, 0, 0, 0, 0, 0, 0]
    return alpha


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
        validFiles = getValidFiles(dPar['files'])
        if not validFiles:
            if len(dPar.files) == 1 and dPar.files[0][-4:] == '.mat':
                updateDataParamsMATLAB(dPar, dPar.files[0])
            else:
                raise Exception('No valid files found')
    elif 'dirs' in dPar:
        dPar.dirs = dPar.dirs.split(',')
        validFiles = getValidFiles(getFiles(dPar.dirs))
        if validFiles:
            updateDataParamsDICOM(dPar, validFiles)
        else:
            raise Exception('No valid files found')
    else:
        raise Exception('No "files" or "dirs" found in dPar config file')


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


def getFattyAcidComposition(rho):
    nFAC = len(rho) - 2 # Number of Fatty Acid Composition Parameters
    eps = sys.float_info.epsilon
    CL, UD, PUD = None, None, None

    if nFAC == 1:
        # UD = F2/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
    elif nFAC == 2:
        # UD = F2/F1
        # PUD = F3/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
    elif nFAC == 3:
        # UD = F2/F1
        # PUD = F3/F1
        # CL = F4/F1
        UD = np.abs(rho[2] / (rho[1] + eps))
        PUD = np.abs(rho[3] / (rho[1] + eps))
        CL = np.abs(rho[4] / (rho[1] + eps))
    else:
        raise Exception('Unknown number of Fatty Acid Composition parameters: {}'.format(nFAC))

    return CL, UD, PUD


# Get total fat component (for Fatty Acid Composition; trivial otherwise)
def getFat(rho, alpha):
    nVxl = np.shape(rho)[1]
    fat = np.zeros(nVxl, dtype=complex)
    for m in range(1, alpha.shape[0]):
        fat += sum(alpha[m, 1:])*rho[m]
    return fat


# Perform fat/water separation and return prescribed output
def reconstruct(dPar, aPar, mPar):

    # Do the fat/water separation
    rho, B0map, R2map = fatWaterSeparation.reconstruct(dPar, aPar, mPar)
    wat = rho[0]
    fat = getFat(rho, mPar.alpha)

    # Prepare prescribed output
    output = {}
    if 'wat' in aPar.output:
        output['wat'] = np.abs(wat)
    if 'fat' in aPar.output:
        output['fat'] = np.abs(fat)
    if 'phi' in aPar.output:
        output['phi'] = np.angle(wat, deg=True) + 180
    if 'ip' in aPar.output: # Calculate synthetic in-phase
        output['ip'] = np.abs(wat+fat)
    if 'op' in aPar.output: # Calculate synthetic opposed-phase
        output['op'] = np.abs(wat-fat)
    if 'ff' in aPar.output: # Calculate the fat fraction
        if aPar.magnDiscr:  # to avoid bias from noise
            output['ff'] = 100 * np.real(fat / (wat + fat + sys.float_info.epsilon))
        else:
            output['ff'] = 100 * np.abs(fat)/(np.abs(wat) + np.abs(fat) + sys.float_info.epsilon)
    if 'B0map' in aPar.output:
        output['B0map'] = B0map
    if 'R2map' in aPar.output:
        output['R2map'] = R2map

    # Do any Fatty Acid Composition in a second pass
    if mPar.nFAC > 0:
        rho = fatWaterSeparation.reconstruct(dPar, aPar.pass2, mPar.pass2, B0map, R2map)[0]
        CL, UD, PUD = getFattyAcidComposition(rho)
    
        if 'CL' in aPar.output:
            output['CL'] = CL
        if 'UD' in aPar.output:
            output['UD'] = UD
        if 'PUD' in aPar.output:
            output['PUD'] = PUD

    return output


# Read configuration file
def readConfig(file, section):
    config = configparser.ConfigParser()
    config.read(file)
    return AttrDict(config[section])


# Wrapper function
def main(dataParamFile, algoParamFile, modelParamFile, outDir=None):
    # Read configuration files
    dPar = readConfig(dataParamFile, 'data parameters')
    aPar = readConfig(algoParamFile, 'algorithm parameters')
    mPar = readConfig(modelParamFile, 'model parameters')

    # Setup configuration objects
    setupDataParams(dPar, outDir)
    setupModelParams(mPar, dPar.clockwisePrecession, dPar.Temperature)
    setupAlgoParams(aPar, dPar.N, mPar.nFAC)

    print('B0 = {}'.format(round(dPar.B0, 2)))
    print('N = {}'.format(dPar.N))
    print('t1/dt = {}/{} msec'.format(round(dPar.t1*1000, 2),
                                      round(dPar.dt*1000, 2)))
    print('nx,ny,nz = {},{},{}'.format(dPar.nx, dPar.ny, dPar.nz))
    print('dx,dy,dz = {},{},{}'.format(
        round(dPar.dx, 2), round(dPar.dy, 2), round(dPar.dz, 2)))

    # Run fat/water processing    
    if aPar.use3D or len(dPar.sliceList) == 1:
        if 'reconSlab' in dPar:
            slabs = getSlabs(dPar.sliceList, dPar.reconSlab)
            for iSlab, (slices, z) in enumerate(slabs):
                print('Processing slab {}/{} (slices {}-{})...'
                      .format(iSlab+1, len(slabs), slices[0]+1, slices[-1]+1))
                slabDataParams = getSlabDataParams(dPar, slices, z)
                output = reconstruct(slabDataParams, aPar, mPar)
                save(output, slabDataParams) # save data slab-wise to save memory
        else:
            output = reconstruct(dPar, aPar, mPar)
            save(output, dPar)
    else:
        output = []
        for z, slice in enumerate(dPar.sliceList):
            print('Processing slice {} ({}/{})...'
                  .format(slice+1, z+1, len(dPar.sliceList)))
            sliceDataParams = getSliceDataParams(dPar, slice, z)
            output.append(reconstruct(sliceDataParams, aPar, mPar))
        save(mergeOutputSlices(output), dPar)


if __name__ == '__main__':
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--dataParamFile', '-d', default='',  type="string",
                 help="Name of data parameter configuration text file")
    p.add_option('--algoParamFile', '-a', default='',  type="string",
                 help="Name of algorithm parameter configuration text file")
    p.add_option('--modelParamFile', '-m', default='',  type="string",
                 help="Name of model parameter configuration text file")

    # Parse command line
    options, arguments = p.parse_args()

    main(options.dataParamFile, options.algoParamFile, options.modelParamFile)