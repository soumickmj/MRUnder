#!/usr/bin/env python

"""
This is the main module, entry point to the pipeline.
All parameters have to be configured here
"""

import glob
import os
from pathlib import Path

import numpy as np
import pydicom
import scipy.io as sio
import torchio as tio
from scipy.signal import resample
from tqdm import tqdm

from CartesianUndersampling.Perform import performUndersampling as cartUnder
from RadialUndersampling.Perform import performUndersampling as radUnder
from Sampler import Sampler
from utils.Coils import generateBirdcageCSM
from utils.HandleDicom import ListRead
from utils.HandleNifti import FileRead, FileSave

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

######Params configuration zone starts here
useExistingMATs = True # [True/False] If an existing MAT file containing the sampling pattern (mask or om) is to be used.
fullySampledPath = r'/run/media/soumick/Enterprise/Datasets/IXI/ISO_Resampled2T2/T1-BET' #Root path containing fully sampled images (NIFTIs: .img, .nii, .nii.gz or DICOMs: .ima, .dcm)
min_scan_no = None # Will be only used for DICOMs. If a single folder contains DICOMs from multiple scans, then using this parameter the starting scan number can be mentioned. If set to None, then will start from the very beginning.
max_scan_no = None #Will be only used for DICOMs. Similar to the last one, itdenotes the last scan that to be considered. If set to None, then scans will be considered till the very last.
underSampledOutPath = r'/run/media/soumick/Enterprise/Datasets/IXI/ISO_Resampled2T2/Under/T1-BET-256' #Root path to store the undersampled output
outFolder = r'Radial30spGA' #Inside the underSampledOutPath, this folder will be created. Inside which the undersampled results will be stored
zeropadOutput = True #By default set to True, when set to False doesn't zero pad the k-Space and decreases the pixel resolution of the output image. This should only be set True when using Cartesian CenterMasks
keepOriginalFormat = False # [True/False] Will be only used for NIFTIs. Specifies whether to keep the original NIFTI extension (e.g. .img) or different file extension to be used while saving
saveFileFormat = '.nii.gz' # File extension to be used while saving the undersampled soutput. For NIFTIs, if keepOriginalFormat=True, then this will be ignored.
nCoilElements = 0 # set it to zero if coil profile not needed

NormWithABS = True #If False then Real will be used, if true then ABS

#Params for using MATs - will be ignored if useExistingMATs is False
isRadial = True
mask_or_om_path = r"/home/soumick/ReadyRoom/GitHub/NCC1701/Cargo/Radial30spGA.mat"

#Params for coil simulation
relative_radius = 0.8 #for birdcage simulation
simulate4each = True #This is needed when they have different height and width. If true, then inputShape varible will be used which is mentioned below
fullySampledCoilImgOutPath = r''#None# r'' #It will only be used when nCoilElements > 0 and this variable is not None. When you don't want to save the fully sampled coil images, then set it to None 

#Params for generating fresh sampling patterns - will be ignored if useExistingMATs is True
recalculateUndersampling4Each = False
staticSamplingFileName =  r'' #To be used if recalculateUndersampling4Each is set to False, to store the generated sampling pattern
inputShape = (256,256) #This is a must have when not recalculating sampling patterns for each volume seperately. If recalculateUndersampling4Each set to True, then this is ignored. Also used when coil is not getting simulated for each
croporpad = True
fullySampledCropPaddedPath = ""#r'/run/media/soumick/Enterprise/Datasets/IXI/ISO_Resampled2T2/T1-BET-256'

undersamplingType = 0 #Cartesian Samplings := 0: Varden1D, 1: Varden2D, 2: Uniform, 3: CenterMaskPercent, 4: CenterMaskIgnoreLines, 5: CenterRatioMask, 6: CenterSquareMask, 7: High-frequency Mask 
                      #Radial Samplings := 10: Golden Angle, 11: Equi-distance (Yet to be implimented)
percentOfKSpace = 0.10 #[between 0 and 1] Percent of k-Space to be sampled. To be used for Cartesian samplings except undersamplingType = 2, 4
stepsize = 4 #[arbitrary] Step size of k-Space sampling lines. To be used by Uniform sampling (Cartesian sampling : 2)
lines2ignore = 10 #[arbitrary] How many lines to ignore from each side of the k-Space. To be used by CenterMaskIgnoreLines sampling (Cartesian sampling : 4)
maxAmplitude4PDF = 0.5 #[between 0 and 1] compression factor of distribution. To be used by Varden1D and High-frequency Mask (Cartesian samplings : 0, 7)
ROdir = 0 #[0, 1, 2 (both-direction)] Read-out direction. To be used by Varden masks, uniform and high-frequency mask (Cartesian samplings : 0, 2, 7)
noOfSpokes = 30 #[arbitrary] Number of spokes to sample. To be used by Radial samplings
fullresSpokesMulFactor = 2 #[arbitrary] Helps to define full resolution during radial sampling (GA), as in theory it can in infinite. Siemens recomands 2 or 3. 
interpolationSize4NUFFT = 6 #To be used by Radial Samplings
sliceUndersamplingFactor = 1 #[arbitrary] For Undersampling in the slice direction, this factor can be used. Setting this to 1 will make it inactive. Setting this more than 1 will choose every Nth slice. 
sliceZPadFourier = False
safeSliceUndersampling = True


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


if not simulate4each :
    csm = generateBirdcageCSM(inputShape, nCoilElements, relative_radius)
else:
    csm = None

def _croppad(fullImgVol, inplane_size, fullpath_file_fully=None):
    if len(fullImgVol.shape) == 3 and len(inplane_size) == 2:
        inplane_size += (fullImgVol.shape[-1],)
    cop = tio.transforms.CropOrPad(inplane_size)
    fullImgVol = cop(np.expand_dims(fullImgVol,axis=0))[0]
    if bool(fullySampledCropPaddedPath) and fullpath_file_fully is not None:
        fullpath_file_cop = fullpath_file_fully.replace(fullySampledPath, fullySampledCropPaddedPath)
        os.makedirs(os.path.dirname(fullpath_file_cop), exist_ok=True)
        FileSave(fullImgVol, fullpath_file_cop)
    return fullImgVol
    
def _getCoilImages(fullImgVol, csm, fullpath_file_fully=None):
    if csm is None:
        csm = generateBirdcageCSM(fullImgVol.shape[0:2], nCoilElements, relative_radius)
    coilVolComplex = np.zeros(fullImgVol.shape+(nCoilElements,), dtype=np.complex64)
    for i in range(fullImgVol.shape[-1]):
        img = fullImgVol[...,i]
        img = img[np.newaxis, :, :] * csm
        coilVolComplex[...,i,:] = img.transpose((1,2,0))
    if fullySampledCoilImgOutPath is not None and fullpath_file_fully is not None:
        fullpath_file_fullycoil = fullpath_file_fully.replace(fullySampledPath, fullySampledCoilImgOutPath)
        os.makedirs(os.path.dirname(fullpath_file_fullycoil), exist_ok=True)
        if NormWithABS:
            coilVol = abs(coilVolComplex)
        else:
            coilVol = coilVolComplex.real
        FileSave(coilVol, fullpath_file_fullycoil)
    return coilVolComplex

def _undersample(fullImgVol, fullpath_file_under):
    try:
        if(not isRadial):
            global mask
        else:
            global om, dcf, interpolationSize4NUFFT
        if recalculateUndersampling4Each:
            samplings = sampler.calculateSamplings(slice=fullImgVol[...,0], returnMeta=True)
            if(not isRadial):
                mask = samplings['mask'] 
                underImgVol = cartUnder(fullImgVol, mask, zeropad=zeropadOutput)
                samplingfilename = fullpath_file_under + '.mask.mat'
            else:
                om = samplings['om'] 
                dcf = samplings['dcf'].squeeze() 
                underImgVol = radUnder(fullImgVol, om, dcf)
                samplingfilename = fullpath_file_under + '.om.mat'
            sio.savemat(samplingfilename, samplings)
        else:
            if len(fullImgVol.shape) == 4:
                underImgVol = np.zeros(fullImgVol.shape, dtype=fullImgVol.dtype)
                for i in range(fullImgVol.shape[3]):
                    coilImgFull = fullImgVol[:,:,:,i] 
                    if(not isRadial):
                        coilImgUnder = cartUnder(coilImgFull, mask, zeropad=zeropadOutput)
                    else:
                        coilImgUnder = radUnder(coilImgFull, om, dcf, interpolationSize4NUFFT)
                    underImgVol[:,:,:,i] = coilImgUnder 
            else:
                if(not isRadial):
                    underImgVol = cartUnder(fullImgVol, mask, zeropad=zeropadOutput)
                else:
                    underImgVol = radUnder(fullImgVol, om, dcf, interpolationSize4NUFFT)
        if NormWithABS:
            underImgVol = abs(underImgVol)
        else:
            underImgVol = underImgVol.real
        underImgVol = underImgVol[:,:,::sliceUndersamplingFactor,...]
        if sliceZPadFourier:
            underImgVol = resample(x=underImgVol, num=fullImgVol.shape[2], axis=2)
                        
        FileSave(underImgVol, fullpath_file_under)
    except Exception as ex:
        print(ex)

#Deal with NIFTI
types = ('.img', '.nii', '.nii.gz') # the tuple of file types
files = []
for type in types:
    files.extend(glob.glob(fullySampledPath+'/**/*'+type, recursive=True))
#files = glob.glob(fullySampledPath+'/**/*.img', recursive=True)

for fullpath_file_fully in tqdm(files):
    fullImgVol = FileRead(fullpath_file_fully).squeeze() #Squeeze to remove channel dim if only one channel
    fullImgVol = _croppad(fullImgVol, inputShape, fullpath_file_fully) if croporpad else fullImgVol
    if safeSliceUndersampling and fullImgVol.shape[-1] % sliceUndersamplingFactor != 0:
        print("Skipping as nSlice not divisable by slice undersampling factor")
        continue

    fullpath_file_under = fullpath_file_fully.replace(fullySampledPath, underSampledOutPath)
    os.makedirs(os.path.dirname(fullpath_file_under), exist_ok=True) #create directorries if doesnt exist
    if not keepOriginalFormat:
        filename, _ = os.path.splitext(fullpath_file_under)
        fullpath_file_under = filename + saveFileFormat
    if nCoilElements != 0:
        fullImgVol = _getCoilImages(fullImgVol, csm, fullpath_file_fully)
    _undersample(fullImgVol, fullpath_file_under)


#Deal with DICOMs
types = ('.ima', '.dcm') # the tuple of file types
files = []
for type in types:
    files.extend(glob.glob(fullySampledPath+'/**/*'+type, recursive=True))

dicoms = {}
for file in tqdm(files):
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

for identifier, files in tqdm(dicoms.items()):
    fullImgVol = ListRead(files).squeeze() #Squeeze to remove channel dim if only one channel
    fullImgVol = _croppad(fullImgVol, inputShape, fullpath_file_fully) if croporpad else fullImgVol
    if safeSliceUndersampling and fullImgVol.shape[-1] % sliceUndersamplingFactor != 0:
        print("Skipping as nSlice not divisable by slice undersampling factor")
        continue

    fullpath_fully = files[0].replace(os.path.basename(files[0]),'')
    fullpath_file_under = fullpath_fully.replace(fullySampledPath, underSampledOutPath) + identifier + saveFileFormat
    os.makedirs(os.path.dirname(fullpath_file_under), exist_ok=True) #create directorries if doesnt exist

    fullImgVol = _croppad(fullImgVol, inputShape, fullpath_file_fully) if croporpad else fullImgVol
    if nCoilElements != 0:
        fullImgVol = _getCoilImages(fullImgVol, csm, fullpath_file_fully)
    _undersample(fullImgVol, fullpath_file_under)
