 # -*- coding: utf-8 -*-
"""
ATOM SIMULATOR Squareatoms
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

from numpy import minimum as minn
from numpy import maximum as maxx 
from scipy.ndimage import gaussian_filter

from hexatoms import hexatoms
from squareatoms import squareatoms
from stripes import stripes


def moirelattice(pix, L, a1, a2, a3, moireBtn, modeBtn, lattice1, lattice2, lattice3, theta_offset, theta_tw, theta_tw2, e11, e12, e22, d11, d12, d22, f11, f12, f22,  alpha1, beta1, alpha2, beta2, alpha3, beta3, eta, xi, origin1, origin2, origin3, filter_bool, sigma,center, strain1_frame = "Local lattice axes", strain2_frame = "Local lattice axes", strain3_frame = "Local lattice axes"):
   
    ### Define the rotation angles

    # Rotate the first lattice (offset rotation of the whole image)
    theta_im = theta_offset 
    
    # Rotate the second lattice wrt to the first one (add theta_offset + theta_twist)
    theta_tw12 = theta_tw + theta_im

    # Rotate the third lattice wrt to the second one 
    theta_tw23 = (theta_tw2 + theta_tw + theta_im)


    ## CREATE FIRST LATTICE ##
    if lattice1 == 'Hexagonal':
        Z1, fftZ1 = hexatoms(pix, L, a1, theta_im, e11, e12, e22, alpha1, beta1, origin1,center, strain1_frame)

    elif lattice1 == 'Square':
        Z1, fftZ1 = squareatoms(pix, L, a1, theta_im, e11, e12, e22,center, strain1_frame)

    elif lattice1 == 'Stripes':
        Z1, fftZ1 = stripes(pix, L, a1, theta_im, e11, e12, e22,center, strain1_frame)


    ## CREATE SECOND LATTICE ##
    if lattice2 == 'Hexagonal':
        Z2, fftZ2 = hexatoms(pix, L, a2, theta_tw12, d11, d12, d22, alpha2, beta2, origin2,center, strain2_frame)
   
    elif lattice2 == 'Square':
        Z2, fftZ2 = squareatoms(pix, L, a2, theta_tw12, d11, d12, d22,center, strain2_frame)

    elif lattice2 == 'Stripes':
        Z2, fftZ2 = stripes(pix, L, a2, theta_tw12, d11, d12, d22,center, strain2_frame)
  

    # If only doing bilayer, create the moire lattice: Z = Z1*Z2 and filter it if filter btn is checked
    if moireBtn == 'Bilayer':
        if modeBtn == 'Simple':
            if eta<0.0:
                eta = 0.0
            elif eta>1.0:
                eta = 1.0

            Z = (eta * Z1 * Z2) + (1-eta)*(Z1 + Z2)

        if modeBtn == 'Log':
            if xi<0.0:
                xi = 0.0
            elif xi>10.0:
                xi = 10.0

            # Add small non-zero constant to avoid -inf from log 0
            Z = np.log(1e-6 + Z1 + Z2*np.exp(-xi))


        if filter_bool == True: 
            Z = gaussian_filter(Z, sigma,mode='mirror')
       
        fftZ = np.abs(npf.fftshift(npf.fft2(Z - np.mean(np.mean(Z)))))

        fftZ = mat2gray(fftZ)

    
    
    ## CREATE THIRD LATTICE IF TRILAYER MOIRE BUTTON IS CHOSEN ## 
    elif moireBtn == 'Trilayer':
        if lattice3 == 'Hexagonal':
            Z3, fftZ3 = hexatoms(pix, L, a3, theta_tw23, f11, f12, f22, alpha3, beta3, origin3,center, strain3_frame)
            
        elif lattice3 == 'Square':
            Z3, fftZ3 = squareatoms(pix, L, a3, theta_tw23, f11, f12, f22,center, strain3_frame)

        elif lattice3 == 'Stripes':
            Z3, fftZ3 = stripes(pix, L, a3, theta_tw23, f11, f12, f22,center, strain3_frame)
        
        if modeBtn == 'Simple':
            if eta<0.0:
                eta = 0.0
            elif eta>1.0:
                eta = 1.0  
      
            Z = (eta * Z1 * Z2 * Z3) + (1-eta)*(Z1 + Z2 + Z3)

        if modeBtn == 'Log':
            if xi<0.0:
                xi = 0.0
            elif xi>10.0:
                xi = 10.0

            Z = np.log(1e-6 + Z1 + Z2*np.exp(-xi) + Z3*np.exp(-2*xi))

      # Low pass filter the image if the button is checked:
        if filter_bool == True: 

            Z = gaussian_filter(Z, sigma,mode='mirror') # filter the stacked 3 lattices

        fftZ = np.abs(npf.fftshift(npf.fft2(Z - np.mean(np.mean(Z)))))


        fftZ = mat2gray(fftZ)
 
    # Normalize the real space image
    Z = mat2gray(Z)


    return Z, np.abs(fftZ)


# explicit function to normalize the 2D matrix.
def mat2gray(Z_un):
    Z = (Z_un - np.min(np.min(Z_un)))/(np.max(np.max(Z_un)) - np.min(np.min(Z_un)))
    return Z


def calculateMoireWavelength(a1, a2, theta_degrees):
    """
    calculates the moire wavelength for two lattice constants
    separated by a relative twist angle

    the returned wavelength is in nanometers
    """

    # numpy trig functions expect angles in radians
    theta_radians = np.radians(theta_degrees)

    denominator = np.sqrt(a1**2 + a2**2 - 2 * a1 * a2 * np.cos(theta_radians))

    # identical and aligned lattices have an infinite moire wavelength
    if np.isclose(denominator, 0):
        return np.inf

    return (a1 * a2) / denominator

def calculateMoireTwistAngle(a1, a2, wavelength):
    """
    calculate the magnitude of the relative twist angle
    from two lattice constants and a moire wavelength

    the returned angle is in degrees
    """

    if a1 <= 0 or a2 <= 0 or wavelength <= 0:
        raise ValueError("Lattice constants and wavelength must be positive.")

    cos_theta = (a1**2 + a2**2 - (a1 * a2 / wavelength)**2) / (2 * a1 * a2)

    # allow for tiny floating point errors near -1 or 1
    if cos_theta < -1 - 1e-12 or cos_theta > 1 + 1e-12:
        raise ValueError("These values do not produce a real twist angle.")

    cos_theta = np.clip(cos_theta, -1, 1)

    return np.degrees(np.arccos(cos_theta))

def calculateMoireLatticeConstant(other_lattice, theta_degrees, wavelength):
    """
    calculate a missing lattice constant from the other lattice
    constant, relative twist angle, and moire wavelength

    returns a tuple containing every positive real solution

    there may be one or two valid solutions
    """

    if other_lattice <= 0 or wavelength <= 0:
        raise ValueError("Lattice constant and wavelength must be positive.")

    theta_radians = np.radians(theta_degrees)
    cos_theta = np.cos(theta_radians)

    # rearranging moire wavelength eqn gives quadratic in terms of unknown lattice constant
    A = wavelength**2 - other_lattice**2
    B = -2 * wavelength**2 * other_lattice * cos_theta
    C = wavelength**2 * other_lattice**2

    # if A is zero the quadratic reduces to a linear equation!
    if np.isclose(A, 0):
        if np.isclose(B, 0):
            raise ValueError("These values do not determine a lattice constant.")

        solution = -C / B

        if solution <= 0:
            raise ValueError("These values do not produce a positive lattice constant.")

        return (solution)

    discriminant = B**2 - 4 * A * C

    # allow for a tiny negative value caused by floating point error
    if discriminant < -1e-12:
        raise ValueError("These values do not produce a real lattice constant.")

    discriminant = max(discriminant, 0)
    square_root = np.sqrt(discriminant)

    solutions = [
        (-B + square_root) / (2 * A),
        (-B - square_root) / (2 * A)
    ]

    # keep only positive solutions and remove duplicates
    positive_solutions = []

    for solution in solutions:
        if solution <= 0:
            continue

        if not any(np.isclose(solution, existing) for existing in positive_solutions):
            positive_solutions.append(solution)

    if not positive_solutions:
        raise ValueError("These values do not produce a positive lattice constant.")

    return tuple(sorted(positive_solutions))

def calculateMixedMoireComponents(lattice1, lattice2, a1, a2, theta_degrees, number_of_components = 3):
    """
    calculate the longest first-shell moire components for
    a pair of different lattice types

    each moire reciprocal vector is the difference between one
    first-shell reciprocal vector from each lattice:

        q = G2 - G1

    the corresponding real-space wavelength is:

        lambda = 2*pi / |q|

    this reports the longest unique first-shell components

    See:
    https://arxiv.org/abs/1001.2798
    """

    if lattice1 == lattice2:
        raise ValueError("Mixed components require different lattice types.")

    supported_lattices = ("Hexagonal", "Square", "Stripes")

    if lattice1 not in supported_lattices:
        raise ValueError("Layer 1 must be Hexagonal, Square, or Stripes.")

    if lattice2 not in supported_lattices:
        raise ValueError("Layer 2 must be Hexagonal, Square, or Stripes.")

    if a1 <= 0 or a2 <= 0:
        raise ValueError("Lattice constants must be positive.")

    if number_of_components < 1:
        raise ValueError("Number of components must be at least one.")

    def createFirstShell(lattice_type, lattice_constant):
        """
        create reciprocal vectors matching the conventions 
        used by hexatoms.py and squareatoms.py and stripes.py
        """

        scale = 2 * np.pi / lattice_constant

        if lattice_type == "Square":
            scale = 2 * np.pi / lattice_constant

            return [
                scale * np.array([1.0, 0.0]),
                scale * np.array([0.0, 1.0]),
                scale * np.array([-1.0, 0.0]),
                scale * np.array([0.0, -1.0])
            ]

        elif lattice_type == "Stripes":
            return [
                scale * np.array([1.0, 0.0]),
                scale * np.array([-1.0, 0.0])
            ]

        k1 = scale * np.array([1.0, 1.0 / np.sqrt(3)])
        k2 = scale * np.array([-1.0, 1.0 / np.sqrt(3)])
        k3 = -(k1 + k2)

        return [k1, k2, k3, -k1, -k2, -k3]

    # keep layer 1 fixed and rotate layer 2 by the relative angle
    theta_radians = np.radians(theta_degrees)

    rotation_matrix = np.array([
        [np.cos(theta_radians), -np.sin(theta_radians)],
        [np.sin(theta_radians),  np.cos(theta_radians)]
    ])

    layer1_vectors = createFirstShell(lattice1, a1)
    layer2_vectors = createFirstShell(lattice2, a2)

    rotated_layer2_vectors = [np.matmul(vector, rotation_matrix) for vector in layer2_vectors]

    components = []

    for vector1 in layer1_vectors:
        for vector2 in rotated_layer2_vectors:
            q_vector = vector2 - vector1

            # find magnitude of q-vector
            q_magnitude = np.linalg.norm(q_vector)

            if np.isclose(q_magnitude, 0):
                continue

            wavelength = 2 * np.pi / q_magnitude

            q_angle = np.degrees(np.arctan2(q_vector[1], q_vector[0]))

            # q and -q describe the same real cosine component 
            # so we store the direction over a 180 degree interval
            q_angle = q_angle % 180

            fringe_angle = (q_angle + 90) % 180

            new_component = {
                "wavelength": wavelength,
                "q_magnitude": q_magnitude,
                "q_angle": q_angle,
                "fringe_angle": fringe_angle
            }

            duplicate = False

            for existing_component in components:
                same_wavelength = np.isclose(wavelength, existing_component["wavelength"], rtol = 1e-9, atol = 1e-12)

                same_direction = np.isclose(q_angle, existing_component["q_angle"], rtol = 0, atol = 1e-9)

                if same_wavelength and same_direction:
                    duplicate = True
                    break

            if not duplicate:
                components.append(new_component)

    # the smallest reciprocal vector difference gives the longest real space wavelength
    components.sort(key = lambda component: component["q_magnitude"])

    return components[:number_of_components]



