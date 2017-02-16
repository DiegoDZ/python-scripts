# -*- coding: utf-8 -*-
"""

This script creates the matrix of covariances, C, and its inverse, H. With H and the mesoscopic and macroscopic profiles, it computes the conjugates variables.

Created on Thu Jul 14 10:32:27 2016

@author: DiegoDZ
"""
import numpy as np
from scipy import linalg


##########################################################
# Compute the matrix of covariances, C, and its inverse, H
##########################################################

# Looking one profile output file: Number of zero rows until the first row with at least one element != 0
number_zeros_lower = 16
# Looking one profile output file: Number of zero rows after the last row with at least one element != 0
number_zeros_upper = 17

# Load files
c00_1 = np.loadtxt('covariance_density')
c00_2 = np.loadtxt('term_density')
c01_1 = np.loadtxt('covariance_densityMesoEnergy')
c01_2 = np.loadtxt('term_energyDensity')
c10_1 = np.loadtxt('covariance_mesoEnergyDensity')
c10_2 = np.loadtxt('term_energyDensity')
c11_1 = np.loadtxt('covariance_mesoEnergy')
c11_2 = np.loadtxt('term_energy')

# Delete zero elements. In this stage we use number_zeros_lower,number_zeros_upper and number_nodes.
#number_nodes = len(c00_1) # All files have the same number of rows, so it is not necessary to compute the lenght of all files
#c00_1 = c00_1[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c00_2 = c00_2[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c01_1 = c01_1[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c01_2 = c01_2[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c10_1 = c10_1[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c10_2 = c10_2[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c11_1 = c11_1[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
#c11_2 = c11_2[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]

# Transform arrays into matrix
C00_1 = np.asmatrix(c00_1)
C00_2 = np.asmatrix(c00_2)
C01_1 = np.asmatrix(c01_1)
C01_2 = np.asmatrix(c01_2)
C10_1 = np.asmatrix(c10_1)
C10_2 = np.asmatrix(c10_2)
C11_1 = np.asmatrix(c11_1)
C11_2 = np.asmatrix(c11_2)

# Calculate the terms C00, C01, C10, C11
C00 = C00_1 - C00_2
C01 = C01_1 - C01_2
C10 = C10_1 - C10_2
C11 = C11_1 - C11_2

# Create the matrix of covariances, C
C = np.bmat([[C00, C01],[C10, C11]])

# Compute the pseudo-inverse of C, H, and the pseudo-inverse of C11, He. 
H = linalg.pinv(C, rcond = 1e-12)
He = linalg.pinv(C11, rcond = 1e-12)

# Save the matrix C, the matrix H
np.savetxt('C-matrix', C)
np.savetxt('H-matrix', H)
np.savetxt('He-matrix', He)

#EOF
