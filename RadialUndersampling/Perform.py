#!/usr/bin/env python

"""
This module performs the undersampling operation, 
for Radial Sampling pattern

"""

import scipy.io as sio
import numpy as np
from pynufft import NUFFT

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def performUndersampling(fullImgVol, om=None, dcf=None, interpolationSize4NUFFT=6, complex2real = np.abs, ommatpath=None):
    #Either send om and dcf, or ommatpath.
    #path will only be used in om not supplied
    if om is None:
        temp_mat =  sio.loadmat(ommatpath)
        om = temp_mat['om']
        dcf = temp_mat['dcf'].squeeze()

    imageSize = fullImgVol.shape[0]
    baseresolution = imageSize*2

    NufftObj = NUFFT()

    Nd = (baseresolution, baseresolution)  # image size
    Kd = (baseresolution*2, baseresolution*2)  # k-space size 
    Jd = (interpolationSize4NUFFT, interpolationSize4NUFFT)  # interpolation size

    NufftObj.plan(om, Nd, Kd, Jd)

    underImgVol = np.zeros(fullImgVol.shape, dtype=fullImgVol.dtype)
    for i in range(fullImgVol.shape[-1]):
        oversam_fully = np.zeros((baseresolution,baseresolution), dtype=fullImgVol.dtype)
        oversam_fully[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2] = fullImgVol[...,i]

        y = NufftObj.forward(oversam_fully)
        y = np.multiply(dcf,y)
        oversam_under = NufftObj.adjoint(y)

        underImgVol[...,i] = complex2real(oversam_under[imageSize//2:imageSize+imageSize//2,imageSize//2:imageSize+imageSize//2])

    return underImgVol