# -*- coding: utf-8 -*-

"""
This script computes the covariance of fluctuations between two matrix.

Example:  density_covariance = <mu*nu> - <mu><nu> = A - B

Author: DiegoDZ
Date: 24 june 2016
Modified: 16 december 2016
run: >> python convariance_mm matrix1 matrix2 > output_file

"""

import numpy as np
import sys

def covariance(arg1, arg2):

    # load files
    datafile1 = np.loadtxt(str(arg1))
    datafile2 = np.loadtxt(str(arg2))
    # define number of nodes
    number_nodes = len(datafile1[0])
    # define number of snapshots
    number_snapshots = len(datafile1)
    # create a 3D array in which we will save information per snapshot
    node_density_snapshot = np.zeros((number_snapshots, number_nodes, number_nodes))

    for i in range(0,number_snapshots):
            # compute the outer product (row x row) in each snapshot and save it.
            node_density_snapshot[i,:,:] = np.outer(datafile1[i], datafile2[i])

    # Compute the first term of the covariance (A)
    A = np.sum(node_density_snapshot, axis = 0) / number_snapshots
    # Sum column elements of the datafile and average the result.
    node_density1 = datafile1.sum(axis = 0) / number_snapshots
    node_density2 = datafile2.sum(axis = 0) / number_snapshots
    # Compute the second term of the covariance (B)
    B = np.outer(node_density1, node_density2)

    covariance = A - B
    return covariance

(covariance) = covariance(sys.argv[1], sys.argv[2])

#For convenience the output will save in matrix format.
aux = ''
for line in covariance:
    for element in line:
        aux = aux + str(element) + ' '
    aux = aux + '\n'

print aux

#EOF
