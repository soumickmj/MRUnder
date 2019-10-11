#!/usr/bin/env python

"""
This module contains utils realted to coil - such as coil simulation, coil combination etc.
For now, only generation of bird card coil sensitivies been added. Other profiles to be added in future.
"""

import numpy as np

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished for now. More features to be added"

def generateBirdcageCSM(matrix_size = 256, number_of_coils = 8, relative_radius = 1.5, normalize=True):

    """ Generates birdcage coil sensitivites.    

    :param matrix_size: size of imaging matrix in pixels (default ``256``)
    :param number_of_coils: Number of simulated coils (default ``8``)
    :param relative_radius: Relative radius of birdcage (default ``1.5``)

    This function is from ismrmrd-python-tools, which is heavily inspired by the mri_birdcage.m Matlab script in
    Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/

    """

    if type(matrix_size) is not tuple:
        matrix_size = (matrix_size,matrix_size)

    out = np.zeros((number_of_coils,)+matrix_size,dtype=np.complex64)

    for c in range(0,number_of_coils):
        coilx = relative_radius*np.cos(c*(2*np.pi/number_of_coils))
        coily = relative_radius*np.sin(c*(2*np.pi/number_of_coils))
        coil_phase = -c*(2*np.pi/number_of_coils)

        for y in range(0,matrix_size[0]):
            y_co = float(y-matrix_size[0]/2)/float(matrix_size[0]/2)-coily

            for x in range(0,matrix_size[1]):
                x_co = float(x-matrix_size[1]/2)/float(matrix_size[1]/2)-coilx
                rr = np.sqrt(x_co**2+y_co**2)
                phi = np.arctan2(x_co, -y_co) + coil_phase
                out[c,y,x] =  (1/rr) * np.exp(1j*phi)
                
    if normalize:
         rss = np.squeeze(np.sqrt(np.sum(abs(out) ** 2, 0)))
         out = out / np.tile(rss,(number_of_coils,1,1))
         
    return out
