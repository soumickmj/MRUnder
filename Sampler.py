#!/usr/bin/env python

"""
This class controls various aspect of creating a sampling pattern.
The constructor of this class accepts various parameters related to the samplings.
calculateSamplings function calculates the actual sampling pattern and returns it.

"""

import numpy as np
from CartesianUndersampling.CenterMask import *
from CartesianUndersampling.UniformMask import *
from CartesianUndersampling.VardenMask import *
from CartesianUndersampling.HighFrequencyMask import *
from RadialUndersampling.GoldenAngle import *


__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"


class Sampler(object):
    """description of class"""

    def __init__(self, undersamplingType, percentOfKSpace, centrePercent, stepsize, lines2ignore, maxAmplitude4PDF, ROdir, noOfSpokes, fullresSpokesMulFactor, interpolationSize, sliceShape=None):
        assert undersamplingType in list(range(0,7+1))+list(range(10,11+1)), 'Unrecognized undersamplingType'
        assert 0 <= percentOfKSpace <= 1, 'Invalid percentOfKSpace'
        assert 0 <= centrePercent <= 1, 'Invalid centrePercent'
        assert 0 <= maxAmplitude4PDF <= 1, 'Invalid maxAmplitude4PDF'
        assert ROdir in [0,1,2], 'Invalid ROdir'

        if undersamplingType == 11: #Equi-distance
            assert False, 'undersamplingType radial equi-distance not yet implimented'

        self.undersamplingType = undersamplingType
        self.percentOfKSpace = percentOfKSpace
        self.centrePercent = centrePercent
        self.stepsize = stepsize
        self.lines2ignore = lines2ignore
        self.maxAmplitude4PDF = maxAmplitude4PDF
        self.ROdir = ROdir
        self.noOfSpokes = noOfSpokes
        self.fullresSpokesMulFactor = fullresSpokesMulFactor
        self.interpolationSize = interpolationSize

        self.MasksWOMetaReturn = [3] #This should contain the list of undersamplingTypes which doesn't return any meta

        if undersamplingType >= 10:
            self.isRadial = True
        else:
            self.isRadial = False

        if sliceShape is None:
            self.dummySlice = None
        else:
            self.dummySlice = np.zeros(sliceShape)

    def calculateSamplings(self, slice=None, returnMeta=False):
        if slice is None:
            assert self.dummySlice is not None, 'If slice is not supplied while calling calculateSamplings, then sliceShape have to be supplied to the constructor of Sampler'
            slice = self.dummySlice
        
        samplings = {}
        if self.undersamplingType == 0: #Varden1D
            data = createVardenMask1D(slice, self.percentOfKSpace, self.maxAmplitude4PDF, self.ROdir, returnMeta)
            metaname = 'PDF'
            samplingname = 'VardenMask1D_percent'+str(self.percentOfKSpace)+'_maxamp'+str(self.maxAmplitude4PDF)+'_rodir'+str(self.ROdir)
        elif self.undersamplingType == 1: #Varden2D
            data = createVardenMask2D(slice, self.percentOfKSpace, self.maxAmplitude4PDF, centrePercent=self.centrePercent, returnPDF=returnMeta)
            metaname = 'PDF'
            samplingname = 'VardenMask2D_percent'+str(self.percentOfKSpace)+'_maxamp'+str(self.maxAmplitude4PDF)+'_centrePercent'+str(self.centrePercent)
        elif self.undersamplingType == 2: #Uniform
            data = createUniformMask(slice, self.stepsize, self.ROdir, returnMeta)
            metaname = 'percentOfKSpace'
            samplingname = 'UniformMask_step'+str(self.stepsize)+'_rodir'+str(self.ROdir)
        elif self.undersamplingType == 3: #CenterMaskPercent
            data = createCenterMaskPercent(slice, self.percentOfKSpace)
            samplingname = 'CenterMaskPercent_percent'+str(self.percentOfKSpace)
        elif self.undersamplingType == 4: #CenterMaskIgnoreLines
            data = createCenterMaskIgnoreLines(slice, self.lines2ignore, returnMeta)
            metaname = 'percentOfKSpace'
            samplingname = 'CenterMaskIgnoreLines_lines2ignore'+str(self.lines2ignore)
        elif self.undersamplingType == 5: #CenterRatioMask
            data = createCenterRatioMask(slice, self.percentOfKSpace, returnMeta)
            metaname = 'NumOfLinesRemoved'
            samplingname = 'CenterRatioMask_percent'+str(self.percentOfKSpace)
        elif self.undersamplingType == 6: #CenterSquareMask
            data = createCenterSquareMask(slice, self.percentOfKSpace, returnMeta)
            metaname = 'NumOfLinesRemoved'
            samplingname = 'CenterSquareMask_percent'+str(self.percentOfKSpace)
        elif self.undersamplingType == 7: #High-frequency Mask
            data = createHighFreqMask(slice, self.percentOfKSpace, self.maxAmplitude4PDF, self.ROdir, returnMeta)
            metaname = 'PDF'
            samplingname = '#HighFreqMask_percent'+str(self.percentOfKSpace)+'_maxamp'+str(self.maxAmplitude4PDF)+'_rodir'+str(self.ROdir)
        elif self.undersamplingType == 8: #Varden2Dv0
            data = createVardenMask2Dv0(slice, self.percentOfKSpace, returnMeta)
            metaname = 'PDF'
            samplingname = 'VardenMask2Dv0_percent'+str(self.percentOfKSpace)
        elif self.undersamplingType == 10: #Golden Angle
            data = {}
            if not bool(self.noOfSpokes):
                baseresolution = slice.shape[0]
                fullspokes = baseresolution * self.fullresSpokesMulFactor
                noOfSpokes = round(fullspokes * self.percentOfKSpace)
                samplingname = 'GoldenAngle_dynspokes'+str(noOfSpokes)+'_percent'+str(self.percentOfKSpace)+'_fulResMulFact'+str(self.fullresSpokesMulFactor)
                omtuple, dcftuple = createGASampling(slice, noOfSpokes, self.fullresSpokesMulFactor, returnFullOM=True, returnInvOM=True)
            else:
                samplingname = 'GoldenAngle_spokes'+str(self.noOfSpokes)+'_fulResMulFact'+str(self.fullresSpokesMulFactor)
                omtuple, dcftuple = createGASampling(slice, self.noOfSpokes, self.fullresSpokesMulFactor, returnFullOM=True, returnInvOM=True)
            data['om'] = omtuple[0]
            data['fullom'] = omtuple[1]
            data['invom'] = omtuple[2]
            data['dcf'] = dcftuple[0]
            data['dcfFullRes'] = dcftuple[1]
            data['dcfInvRes'] = dcftuple[2]
        elif self.undersamplingType == 11: #Equi-distance
            print('TODO')
        
        if self.isRadial:
            samplings = data
        else:
            if returnMeta and self.undersamplingType not in self.MasksWOMetaReturn:
                samplings = {'mask': data[0], metaname: data[1]}
            else:
                samplings = {'mask': data}
        samplings['samplingname'] = samplingname

        return samplings