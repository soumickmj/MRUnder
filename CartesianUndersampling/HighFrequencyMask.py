#!/usr/bin/env python

"""
This module generates High-frequency mask

"""

import math
import numpy as np

__author__ = "Mariio Breitkopf, Soumick Chatterjee"
__copyright__ = "Copyright 2019, Mario Breitkopf, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Mariio Breitkopf", "Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


def createHighFreqMask(slice, percent, compressFactOfDist, ROdir, returnPDF=False):
    mask = np.zeros(slice.shape)
    if ROdir == 2:
        percent = percent/2
        if slice.shape[0] == slice.shape[1]:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, compressFactOfDist, 0)
            mask, _, _ = _mask1DForROdir(mask, percent, compressFactOfDist, 1, distfunc, randseed)
        elif slice.shape[0] > slice.shape[1]:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, compressFactOfDist, 0)
            dim_difference = slice.shape[0] - slice.shape[1]
            _distfunc = distfunc[dim_difference//2:(slice.shape[1]+dim_difference//2)]
            _randseed = randseed[dim_difference//2:(slice.shape[1]+dim_difference//2)]
            mask, _, _ = _mask1DForROdir(mask, percent, compressFactOfDist, 1, _distfunc, _randseed)
        else:
            mask, distfunc, randseed = _mask1DForROdir(mask, percent, compressFactOfDist, 1)
            dim_difference = slice.shape[1] - slice.shape[0]
            _distfunc = distfunc[dim_difference//2:(slice.shape[0]+dim_difference//2)]
            _randseed = randseed[dim_difference//2:(slice.shape[0]+dim_difference//2)]
            mask, _, _ = _mask1DForROdir(mask, percent, compressFactOfDist, 0, _distfunc, _randseed)
    else:
        mask, distfunc, _ = _mask1DForROdir(mask, percent, compressFactOfDist, ROdir)

    if returnPDF:
        if slice.shape[0] > slice.shape[1]:
            return mask, np.tile(distfunc,(slice.shape[1],1))
        else:
            return mask, np.tile(distfunc,(slice.shape[0],1))
    else:
        return mask

def _mask1DForROdir(mask, percent, compressFactOfDist, ROdir, distfunc=None, randseed=None):
    shape = mask.shape[ROdir]
    if distfunc is None or randseed is None:
        #Random Numbers Seed
        randseed = np.random.random(shape)

        #Initialize variables
        x = np.array(range(-math.floor(shape/2)+1,math.floor(shape/2)+1,1))
        xm = math.ceil(x.size/2)
        #mu = 0.5 ;

        currentPercent=1
        while currentPercent > percent:
            #Distribution function
            distfunc = compressFactOfDist*(np.power(x,2) / np.power(xm,2)) 
            #distfunc = np.sqrt(np.power(xm,2)-compressFactOfDist*np.power(xm,2)) / np.sqrt(np.power(xm,2)-compressFactOfDist*np.power(x,2))

            #Selection of k-space lines
            B = np.zeros(shape)
            #B[randseed>distfunc] = 0
            B[randseed<distfunc] = 1
            B[round(shape/2-shape/round(shape/4)):round(shape/2+shape/round(shape/4))] = 1
            currentPercent = np.count_nonzero(B)/B.size
            compressFactOfDist = np.floor(0.95*compressFactOfDist)
    else:
        #Selection of k-space lines
        B = np.zeros(shape)
        #B[randseed>distfunc] = 0
        B[randseed<distfunc] = 1
        B[round(shape/2-shape/round(shape/4)):round(shape/2+shape/round(shape/4))] = 1

    if ROdir == 0:
        mask[B==1,:] = 1
    else: #ROdir == 1:
        mask[:,B==1] = 1

    return mask, distfunc, randseed
