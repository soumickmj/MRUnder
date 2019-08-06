#!/usr/bin/env python

"""
This module helps to handle NIFTI files
Can read individual files but not complete folder
Also, helps the them to be converted to 2D or even 1D, and also back to 3D

"""

import tkinter as tk
from tkinter import filedialog
import numpy as np
import nibabel as nib

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2018, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Finished"

def ChooseFileNRead(Read3D=True):
    """Choose a NIFTI file using file dialog box and read it as an array
    Using: NiBabel""" 
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if(Read3D):
        return FileRead3D(file_path)
    else:
        return FileRead2D(file_path)

def FileRead(file_path, expand_last_dim=False):
    """Read a NIFTI file (3D) using given file path as an array
    Using: NiBabel"""
    nii = nib.load(file_path)
    data = nii.get_data()
    if expand_last_dim: #If channel data not present
        data = np.expand_dims(data, -1)
    return data

def FileRead3D(file_path):
    """Read a NIFTI file (3D) using given file path as an array
    Using: NiBabel"""
    nii = nib.load(file_path)
    data = nii.get_data()
    if (np.shape(np.shape(data))[0] == 3): #If channel data not present
        data = np.expand_dims(data, 3)
    return data

def FileRead2D(file_path):
    """Read a NIFTI file (2D) using given file path as an array
    Using: NiBabel"""
    nii = nib.load(file_path)
    data = nii.get_data()
    if (np.shape(np.shape(data))[0] == 2): #If channel data not present
        data = np.expand_dims(data, 2)
    return data

def ChooseFileNSave(data):
    """Choose a NIFTI file using file dialog box to save it
    Using: NiBabel"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename()
    FileSave(data, file_path)

def FileSave(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if(np.iscomplex(data).any()):
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4)) 
    nib.save(nii, file_path)
    
def Nifti3Dto2D(Nifti3D):
    """Converts From 3D NIFTI to 2D
    Preserves channel info"""
    Nifti2D = Nifti3D.reshape(np.shape(Nifti3D)[0], np.shape(Nifti3D)[1] * np.shape(Nifti3D)[2], np.shape(Nifti3D)[3])
    return Nifti2D

def Nifti2Dto3D(Nifti2D):
    """Converts From 2D NIFTI to 3D
    Preserves channel info"""
    Nifti3D = Nifti2D.reshape(np.shape(Nifti2D)[0],np.shape(Nifti2D)[0],int(np.shape(Nifti2D)[1]/np.shape(Nifti2D)[0]), np.shape(Nifti2D)[2])
    return Nifti3D

def Nifti2Dto1D(Nifti2D): 
    """Converts From 2D NIFTI to 1D
    No Sperate channel info left. It's now all merged together"""
    Nifti1D = Nifti2D.reshape(np.shape(Nifti2D)[0] * np.shape(Nifti2D)[1] * np.shape(Nifti2D)[2])
    return Nifti1D

def Nifti1Dto2D(Nifti1D, height, n_channel):
    """Converts From 1D NIFTI to 2D
    No of Channel introduced seperately"""
    Nifti2D = Nifti1D.reshape(height,int((np.shape(Nifti1D)[0]/height)/n_channel), n_channel)
    return Nifti2D