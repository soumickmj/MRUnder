#!/usr/bin/env python

"""
This module generates radial sampling pattern following Golden Angle Scheme

"""

import math
import numpy as np
from RadialUndersampling.dcf import generateDCF

__author__ = "Mariio Breitkopf, Soumick Chatterjee"
__copyright__ = "Copyright 2019, Mario Breitkopf, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Mariio Breitkopf", "Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def createGASampling(slice, noOfSpokes, fullresSpokesMulFactor=2, returnFullOM=True, returnInvOM=True):
    baseresolution = slice.shape[0]
    fullspokes = baseresolution * fullresSpokesMulFactor
    baseresolution = baseresolution * 2

    fullom = np.zeros((baseresolution*fullspokes,2))
    i=0
    inc = 111.246117975
    for s in range(fullspokes):
        b = math.pi*math.cos(np.deg2rad(np.mod(inc*s,360)))
        a = math.pi*math.sin(np.deg2rad(np.mod(inc*s,360)))
        x = np.linspace(-a, a, baseresolution)
        y = np.linspace(-b, b, baseresolution)
        fullom[i*baseresolution:baseresolution+i*baseresolution,0] = x
        fullom[i*baseresolution:baseresolution+i*baseresolution,1] = y
        i = i+1

    under_om_element = baseresolution * noOfSpokes
    om = fullom[0:under_om_element,:]
    dcf = generateDCF(noOfSpokes, baseresolution)

    omtuple = (om,)
    dcftuple = (dcf,)

    if returnFullOM:
        dcfFullRes = generateDCF(fullspokes, baseresolution)
        omtuple += (fullom,)
        dcftuple += (dcfFullRes,)

    if returnInvOM:
        invom = fullom[under_om_element:,:]
        dcfInvRes = generateDCF(fullspokes-noOfSpokes, baseresolution)
        omtuple += (invom,)
        dcftuple += (dcfInvRes,)

    return omtuple, dcftuple