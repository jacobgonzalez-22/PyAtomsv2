 # -*- coding: utf-8 -*-
"""
ATOM SIMULATOR squareatoms
Created on Mon Nov 15 14:45:06 2021
@author: Asari
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


def squareatoms(pix, L, a, theta, e11, e12, e22, center, strain_frame = "Local lattice axes"):
    """
    Simulates a square atomic lattice and takes the FFT of the lattice

    Creates a (pix x pix) image array of a square atomic lattice with periodicity 'a' (nm),
    rotated from the x-direction by an angle theta. The image has side length L (nm).
    If honeycomb == 1, it creates the same image but with honeycomb pattern. 
    For either choice, the image height is normalized to [0,1].

    Args:

    pix [int]: # of pixels, like 256
    L [float]: length of simulated image in nanometers
    a [float]: atomic lattice constant in nanometers
    theta [degrees]: angle to rotate the image
    e11, e12, e22 [float]: strain tensor elements
    honeycomb [0 or 1]: is asking whether you want the atoms as dots (0) or holes (1)
    origin is optional, you can give a lattice vector [o1,o2]
        where o1,o2 are pixel positions of your origin
    """
  

    # Create a meshgrid of 'pix' number of points, values from 0 --> L
    xx = np.arange(-(pix//2),(pix-1)//2 + 1)*L/(pix-1) 
    [X,Y] = np.meshgrid(xx,xx)
    
    # Create square lattice
    Z = evaluateSquareLatticeAtCoords(X, Y, a, theta, e11, e12, e22, center, strain_frame)

    # FFT of lattice:
    fftZ = np.abs(npf.fftshift(npf.fft2(Z- np.mean(np.mean(Z)))))  # subtract by mean(mean(Z)) to remove the strong peak at k=0 (DC/constant background)
    fftZ = mat2gray(fftZ)

    return Z, fftZ

def evaluateSquareLatticeAtCoords(X, Y, a, theta, e11, e12, e22, center, strain_frame = "Local lattice axes"):
    """
    evaluate a square lattice directly at coordinates x, y
    
    mirrors logic in squareatoms
    """

    X0 = center[0]
    Y0 = center[1] 

    # Reciprocal lattice vectors for square crystal WITH STRAIN
    # k1 = (2*np.pi/a)*np.array([1 + e11, e12])
    # k2 = (2*np.pi/a)*np.array([e12, 1 + e22]) 

    k1_unstrained = (2*np.pi/a)*np.array([1, 0])
    k2_unstrained = (2*np.pi/a)*np.array([0, 1])

    # TESTING:
    strain_tensor = np.array([[e11, e12],
                             [e12, e22]])
    
    reciprocal_strain = np.linalg.inv(np.eye(2) + strain_tensor)

    k1_strained = np.matmul(k1_unstrained, reciprocal_strain)
    k2_strained = np.matmul(k2_unstrained, reciprocal_strain)

    # original
    # k1_strained = (2*np.pi/a)*np.array([1 - e11, -e12])
    # k2_strained = (2*np.pi/a)*np.array([-e12, 1 - e22])

    ## Rotate the lattice by theta (in degrees)
    # Convert theta to radians
    theta_rad = np.deg2rad(theta) 

    # Create rotation matrix to multiply reciprocal lattice vectors to rotate the image 
    rotmat = np.array([[cos(theta_rad), -sin(theta_rad)], 
                       [sin(theta_rad), cos(theta_rad)]])

    if strain_frame == "Global image axes":
        # rotate first then apply strain along global x/y
        k1 = np.matmul(k1_unstrained, rotmat)
        k2 = np.matmul(k2_unstrained, rotmat)

        # TESTING:
        k1 = np.matmul(k1, reciprocal_strain)
        k2 = np.matmul(k2, reciprocal_strain)

        # original
        # k1 = np.array([k1[0] - e11*k1[0] - e12*k1[1],
        #                k1[1] - e12*k1[0] - e22*k1[1]])

        # k2 = np.array([k2[0] - e11*k2[0] - e12*k2[1],
        #                k2[1] - e12*k2[0] - e22*k2[1]])
        
    else:
        # strain along local axes then rotate
        k1 = np.matmul(k1_strained, rotmat)
        k2 = np.matmul(k2_strained, rotmat)

    
    k1x, k1y = k1[0], k1[1]
    k2x, k2y = k2[0], k2[1]
    
    # create unnormalized square lattice
    Z_un = (
        np.cos(k1x*(X-X0) + k1y*(Y-Y0)) +
        np.cos(k2x*(X-X0) + k2y*(Y-Y0))
    )

    # normalize image so values are between 0 and 1
    Z = mat2gray(Z_un)

    return Z

# explicit function to normalize the 2D maatrix
def mat2gray(Z_un):
    Z_min = np.min(Z_un)
    Z_max = np.max(Z_un)

    if Z_max == Z_min:
        return np.zeros_like(Z_un)

    return (Z_un - Z_min) / (Z_max - Z_min)