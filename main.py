#!/usr/bin/env python

"""
This is the main module, entry point to the pipeline.
All parameters have to be configured here
"""

import scipy.io as sio
import os
import glob
from pathlib import Path
import pydicom
from utils.HandleNifti import FileRead, FileSave
from utils.HandleDicom import ListRead
from CartesianUndersampling.Perform import performUndersampling as cartUnder
from RadialUndersampling.Perform import performUndersampling as radUnder
from Sampler import Sampler

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

######Params configuration zone starts here
useExistingMATs = False # [True/False] If an existing MAT file containing the sampling pattern (mask or om) is to be used.
fullySampledPath = r'' #Root path containing fully sampled images (NIFTIs: .img, .nii, .nii.gz or DICOMs: .ima, .dcm)
min_scan_no = None # Will be only used for DICOMs. If a single folder contains DICOMs from multiple scans, then using this parameter the starting scan number can be mentioned. If set to None, then will start from the very beginning.
max_scan_no = None #Will be only used for DICOMs. Similar to the last one, itdenotes the last scan that to be considered. If set to None, then scans will be considered till the very last.
underSampledOutPath = r'' #Root path to store the undersampled output
outFolder = r'' #Inside the underSampledOutPath, this folder will be created. Inside which the undersampled results will be stored
keepOriginalFormat = True # [True/False] Will be only used for NIFTIs. Specifies whether to keep the original NIFTI extension (e.g. .img) or different file extension to be used while saving
saveFileFormat = '.nii.gz' # File extension to be used while saving the undersampled soutput. For NIFTIs, if keepOriginalFormat=True, then this will be ignored.

#Params for using MATs - will be ignored if useExistingMATs is False
isRadial = False
mask_or_om_path = r''

#Params for generating fresh sampling patterns - will be ignored if useExistingMATs is True
recalculateUndersampling4Each = True
staticSamplingFileName =  r'' #To be used if recalculateUndersampling4Each is set to False, to store the generated sampling pattern
inputShape = (256,256) #This is a must have when not recalculating sampling patterns for each volume seperately. If recalculateUndersampling4Each set to True, then this is ignored

undersamplingType = 0 #Cartesian Samplings := 0: Varden1D, 1: Varden2D, 2: Uniform, 3: CenterMaskPercent, 4: CenterMaskIgnoreLines, 5: CenterRatioMask, 6: CenterSquareMask, 7: High-frequency Mask 
                      #Radial Samplings := 10: Golden Angle, 11: Equi-distance (Yet to be implimented)
percentOfKSpace = 0.3 #[between 0 and 1] Percent of k-Space to be sampled. To be used for Cartesian samplings except undersamplingType = 2, 4
stepsize = 4 #[arbitrary] Step size of k-Space sampling lines. To be used by Uniform sampling (Cartesian sampling : 2)
lines2ignore = 10 #[arbitrary] How many lines to ignore from each side of the k-Space. To be used by CenterMaskIgnoreLines sampling (Cartesian sampling : 4)
maxAmplitude4PDF = 0.5 #[between 0 and 1] compression factor of distribution. To be used by Varden1D and High-frequency Mask (Cartesian samplings : 0, 7)
ROdir = 0 #[0, 1, 2 (both-direction)] Read-out direction. To be used by Varden masks, uniform and high-frequency mask (Cartesian samplings : 0, 2, 7)
noOfSpokes = 30 #[arbitrary] Number of spokes to sample. To be used by Radial samplings
fullresSpokesMulFactor = 2 #[arbitrary] Helps to define full resolution during radial sampling (GA), as in theory it can in infinite. Siemens recomands 2 or 3. 
interpolationSize4NUFFT = 6 #To be used by Radial Samplings

######Params configuration zone ends here

underSampledOutPath = os.path.join(underSampledOutPath, outFolder)

if useExistingMATs:
    if(not isRadial):
        mask = sio.loadmat(mask_or_om_path)['mask']
    else:
        temp_mat =  sio.loadmat(mask_or_om_path)
        om = temp_mat['om']
        dcf = temp_mat['dcf'].squeeze()
else:
    if recalculateUndersampling4Each:
        inputShape = None

    sampler = Sampler(undersamplingType, percentOfKSpace, stepsize, lines2ignore, maxAmplitude4PDF, ROdir, noOfSpokes, fullresSpokesMulFactor, interpolationSize4NUFFT, inputShape)
    isRadial = sampler.isRadial

    if not recalculateUndersampling4Each:
        samplings = sampler.calculateSamplings(returnMeta=True)
        if(not isRadial):
            mask = samplings['mask']
        else:
            om = samplings['om']
            dcf = samplings['dcf'].squeeze()
        sio.savemat(staticSamplingFileName, samplings)


def _undersample(fullImgVol, fullpath_file_under):
    if recalculateUndersampling4Each:
        samplings = sampler.calculateSamplings(slice=fullImgVol[...,0], returnMeta=True)
        if(not isRadial):
            mask = samplings['mask'] 
            underImgVol = cartUnder(fullImgVol, mask)
            samplingfilename = fullpath_file_under + '.mask.mat'
        else:
            om = samplings['om'] 
            dcf = samplings['dcf'].squeeze() 
            underImgVol = radUnder(fullImgVol, om, dcf)
            samplingfilename = fullpath_file_under + '.om.mat'
        sio.savemat(samplingfilename, samplings)
    else:
        if(not isRadial):
            underImgVol = cartUnder(fullImgVol, mask)
        else:
            underImgVol = radUnder(fullImgVol, om, dcf, interpolationSize4NUFFT)
    FileSave(underImgVol, fullpath_file_under)

#Deal with NIFTI
types = ('.img', '.nii', '.nii.gz') # the tuple of file types
files = []
for type in types:
    files.extend(glob.glob(fullySampledPath+'/**/*'+type, recursive=True))
#files = glob.glob(fullySampledPath+'/**/*.img', recursive=True)

for fullpath_file_fully in files:
    print(fullpath_file_fully)
    fullpath_file_under = fullpath_file_fully.replace(fullySampledPath, underSampledOutPath)
    os.makedirs(os.path.dirname(fullpath_file_under), exist_ok=True) #create directorries if doesnt exist
    if not keepOriginalFormat:
        filename, _ = os.path.splitext(fullpath_file_under)
        fullpath_file_under = filename + saveFileFormat

    fullImgVol = FileRead(fullpath_file_fully).squeeze() #Squeeze to remove channel dim if only one channel
    _undersample(fullImgVol, fullpath_file_under)


#Deal with DICOMs
types = ('.ima', '.dcm') # the tuple of file types
files = []
for type in types:
    files.extend(glob.glob(fullySampledPath+'/**/*'+type, recursive=True))

dicoms = {}
for file in files:
    #file_name = Path(file).stem
    dicom = pydicom.dcmread(file)
    scan_no = int(dicom.SeriesNumber)
    if (((max_scan_no is None) and (min_scan_no is not None) and (scan_no < min_scan_no))
       or ((max_scan_no is not None) and (min_scan_no is not None) and ((scan_no < min_scan_no) or (scan_no > max_scan_no)))
          or ((max_scan_no is not None) and (min_scan_no is None) and (scan_no > max_scan_no))):
        continue
    dicom_identifier = str(scan_no) + '.' + dicom.ProtocolName + '.' +  str(dicom.SeriesDate) + '.' + str(dicom.SeriesTime)
    if dicom_identifier in dicoms:
        dicoms[dicom_identifier].append(file)
    else:
        dicoms[dicom_identifier] = [file]

for identifier, files in dicoms.items():
    print(identifier)
    fullpath_fully = files[0].replace(os.path.basename(files[0]),'')
    fullpath_file_under = fullpath_fully.replace(fullySampledPath, underSampledOutPath) + identifier + saveFileFormat
    os.makedirs(os.path.dirname(fullpath_file_under), exist_ok=True) #create directorries if doesnt exist

    fullImgVol = ListRead(files).squeeze() #Squeeze to remove channel dim if only one channel
    _undersample(fullImgVol, fullpath_file_under)