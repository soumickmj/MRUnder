#!/usr/bin/env python

"""
This module generates various k-Space center masks:-
Undersampling codes:-
3 : Center of K-Space Mask (Ignore same no of lines both sides, based on given percentage) \\
4 : Center of K-Space Mask (Ignores specified number of of lines both sides)\\
5 : Center of K-Space Mask (Ignores lines based on given percentage, preserving the aspect ratio)\\
6 : Center of K-Space Square Mask (Ignores lines based on given percentage, end mask will have same height and width)\\

"""

import numpy as np

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def createCenterMaskPercent(slice, percent):
    mask = np.ones(slice.shape)
    i = 0
    currentPercent = 1

    while currentPercent > percent:
        i += 1
        mask[0:i, :] = 0
        mask[slice.shape[0]-i:, :] = 0
        mask[:, 0:i] = 0
        mask[:, slice.shape[1]-i:] = 0

        currentPercent = np.count_nonzero(mask)/mask.size

    return mask

def createCenterMaskIgnoreLines(slice, lines2ignore, returnPercent=False):
    #lines2ignore from each side
    mask = np.ones(slice.shape)
    mask[0:lines2ignore, :] = 0
    mask[slice.shape[0]-lines2ignore:, :] = 0
    mask[:, 0:lines2ignore] = 0
    mask[:, slice.shape[1]-lines2ignore:] = 0

    if returnPercent:
        return mask, np.count_nonzero(mask)/mask.size
    else:
        return mask

def createCenterRatioMask(slice, percent, returnNumLinesRemoved=False):
    dim1 = slice.shape[0]
    dim2 = slice.shape[1]
    ratio = dim2/dim1
    
    mask = np.ones(slice.shape)
    dim1_now = dim1
    dim2_should = dim2
    i = 0
    currentPercent = 1

    while currentPercent > percent:
        i += 1
        mask[0:i, :] = 0
        mask[slice.shape[0]-i:, :] = 0

        dim1_now = dim1 - (i*2)
        dim2_should = round(dim1_now * ratio)
        dim2_removal = int((dim2-dim2_should)/2)

        mask[:, 0:dim2_removal] = 0
        mask[:, slice.shape[1]-dim2_removal:] = 0

        currentPercent = np.count_nonzero(mask)/mask.size

   
    if returnNumLinesRemoved:
        linesRemoved_dim1 = dim1 - dim1_now
        linesRemoved_dim2 = dim2 - dim2_should
        return mask, (linesRemoved_dim1, linesRemoved_dim2)
    else:
        return mask

def createCenterSquareMask(slice, percent, returnNumLinesRemoved=False):
    dim1 = slice.shape[0]
    dim2 = slice.shape[1]
    
    mask = np.ones(slice.shape)

    if(dim1 > dim2):
        linesRemoved_dim1 = dim1 - dim2
        linesRemoved_dim2 = 0
        mask[0:int(linesRemoved_dim1/2), :] = 0
        mask[slice.shape[0]-int(linesRemoved_dim1/2):, :] = 0
    else:
        linesRemoved_dim1 = 0
        linesRemoved_dim2 = dim2 - dim1
        mask[:, 0:int(linesRemoved_dim2/2)] = 0
        mask[:, slice.shape[1]-int(linesRemoved_dim2/2):] = 0  


    i = 0
    currentPercent = np.count_nonzero(mask)/mask.size

    while currentPercent > percent:
        i += 1
        mask[0:i+int(linesRemoved_dim1/2), :] = 0
        mask[slice.shape[0]-(i+int(linesRemoved_dim1/2)):, :] = 0
        mask[:, 0:i+int(linesRemoved_dim2/2)] = 0
        mask[:, slice.shape[1]-(i+int(linesRemoved_dim2/2)):] = 0

        currentPercent = np.count_nonzero(mask)/mask.size

    linesRemoved_dim1 = linesRemoved_dim1 + (i*2)
    linesRemoved_dim2 = linesRemoved_dim2 + (i*2)
   
    if returnNumLinesRemoved:
        return mask, (linesRemoved_dim1, linesRemoved_dim2)
    else:
        return mask