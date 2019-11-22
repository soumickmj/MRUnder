#!/usr/bin/env python

"""
This module performs the undersampling operation, 
for Cartesian Sampling pattern

"""

import scipy.io as sio
import numpy as np
from utils.FrequencyTransforms import fft2c, ifft2c

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def performUndersampling(fullImgVol, mask=None, maskmatpath=None, zeropad=True):
    #Either send mask, or maskmatpath.
    #path will only be used in mask not supplied
    fullKSPVol = fft2c(fullImgVol)
    underKSPVol = performUndersamplingKSP(fullKSPVol, mask, maskmatpath,zeropad)
    underImgVol = ifft2c(underKSPVol)
    return underImgVol

def performUndersamplingKSP(fullKSPVol, mask=None, maskmatpath=None, zeropad=True):
    #Either send mask, or maskmatpath.
    #path will only be used in mask not supplied
    if mask is None:
        mask = sio.loadmat(maskmatpath)['mask']
    if zeropad:
        underKSPVol = np.multiply(fullKSPVol.transpose((2,0,1)), mask).transpose((1,2,0))
    else:
        temp = []
        for i in range(mask.shape[0]):
            maskline = mask[i,:]
            if maskline.any():
                temp.append(fullKSPVol[i,...])
        temp = np.array(temp)
        underKSPVol = []
        for i in range(mask.shape[1]):
            maskline = mask[:,i]
            if maskline.any():
                underKSPVol.append(temp[:,i,...])
        underKSPVol = np.array(underKSPVol).swapaxes(0,1)
    return underKSPVol