#!/usr/bin/env python

"""
This module creates density compensation function for radial samplings
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

def generateDCF(spokes, baseresolution):
    dcfRow=np.ones((baseresolution,1))
    for i in range(baseresolution):
       dcfRow[i]=np.abs(baseresolution/2-(i-0.5))
    dcfRow=math.pi/spokes*dcfRow
    dcf=np.tile(dcfRow,(spokes,1)).transpose()
    return dcf