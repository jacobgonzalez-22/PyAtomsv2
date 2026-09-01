 # -*- coding: utf-8 -*-
"""
ATOM SIMULATOR stripes
Created on Fri August 28 18:31:34 2026
@author: Jacob
"""

import numpy as np

from numpy import fft as npf
from numpy import cos as cos
from numpy import sin as sin
from numpy import tan as tan
from numpy import arcsin as arcsin
from numpy import arccos as arccos
from numpy import arctan as arctan

from numpy import sqrt as sqrt
from numpy import log as log
from numpy import pi as pi


def stripes(pix, L, a, theta, e11, e12, e22, center, strain_frame = "Local lattice axes"):
    """
    Simulates a one-dimensional periodic stripe modulation and takes its FFT.

    Creates a (pix x pix) image array with periodicity 'a' (nm).
    The modulation is defined by a single reciprocal-space wavevector and
    can be rotated by theta. The image height is normalized to [0, 1].

    Args:
    pix [int]: number of pixels, e.g. 256
    L [float]: lenght of simulated image in nanometers
    a [float]: stripe periodicity/wavelength in nanometers
    theta [degrees]: angle used to rotate the modulation
    e11, e12, e22 [float]: strain tensor elements
    center: [X0, Y0] center/offset of the modulation
    strain_frame [str]: "Local lattice axes" or "Global lattice axes"
    """
  

    # Create a meshgrid of 'pix' number of points, values from 0 --> L
    xx = np.arange(-(pix//2),(pix-1)//2 + 1)*L/(pix-1) 
    [X,Y] = np.meshgrid(xx,xx)
    
    # Create stripe modulation
    Z = evaluateStripesAtCoords(X, Y, a, theta, e11, e12, e22, center, strain_frame)

    # FFT of modulation:
    fftZ = np.abs(npf.fftshift(npf.fft2(Z- np.mean(np.mean(Z)))))  # subtract by mean(mean(Z)) to remove the strong peak at k=0 (DC/constant background)
    fftZ = mat2gray(fftZ)


    return Z, fftZ

def evaluateStripesAtCoords(X, Y, a, theta, e11, e12, e22, center, strain_frame = "Local lattice axes"):
    """
    Evaluates the stripe modulation directly at coordinates X and Y.

    This mirrors the coordinate-evaluation approach used by hexatoms and squareatoms.
    """

    X0 = center[0]
    Y0 = center[1] 

    # single reciprocal-space wavevector for a 1D modulation
    k_unstrained = (2*np.pi/a)*np.array([1, 0])

    strain_tensor = np.array([[e11, e12],
                             [e12, e22]])
    
    reciprocal_strain = np.linalg.inv(np.eye(2) + strain_tensor)

    k_strained = np.matmul(k_unstrained, reciprocal_strain)


    ## Rotate the lattice by theta (in degrees)
    # Convert theta to radians
    theta_rad = np.deg2rad(theta) 

    # Create rotation matrix to multiply reciprocal lattice vectors to rotate the image 
    rotmat = np.array([[cos(theta_rad), -sin(theta_rad)], 
                       [sin(theta_rad), cos(theta_rad)]])

    if strain_frame == "Global image axes":
        # rotate first then apply strain along global x/y
        k = np.matmul(k_unstrained, rotmat)
        k = np.matmul(k, reciprocal_strain)

    else:
        # strain along local axes then rotate
        k = np.matmul(k_strained, rotmat)

    
    kx, ky = k[0], k[1]

    # create unnormalized one-dimensional periodic modulation
    Z_un = np.cos(kx*(X-X0) + ky*(Y-Y0))

    # normalize image to [0, 1]
    Z = mat2gray(Z_un)

    return Z

# explicit function to normalize the 2D maatrix
def mat2gray(Z_un):
    Z_min = np.min(Z_un)
    Z_max = np.max(Z_un)

    if Z_max == Z_min:
        return np.zeros_like(Z_un)

    return (Z_un - Z_min) / (Z_max - Z_min)
