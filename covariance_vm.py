# -*- coding: utf-8 -*-

"""
This script computes the covariance of fluctuations between a vector and a matrix.
The output is a vector.

Example:  covariance_centerofmass = <mu*nu> - <mu><nu>

Author: DiegoDZ
Date: 29 june 2016
Modified: 15 december 2016

run: >> python convariance_vm.py vector matrix > output_file

"""
import numpy as np
import sys

def covariance(arg1, arg2):

    # load files
    datafile1 = np.loadtxt(str(arg1))
    datafile2 = np.loadtxt(str(arg2))
    # define number nodes
    number_nodes = len(datafile2[0])
    # define number snapshots
    number_snapshots = len(datafile2)
    # create an array in which we will be the output.
    covariance= np.zeros(number_nodes)

    for i in range(0,number_nodes):
        covariance[i] = np.sum(datafile1.T[0,:] * datafile2[:,i]) / number_snapshots - np.sum(datafile1.T[0,:]) * np.sum(datafile2[:,i]) / number_snapshots ** 2

    return covariance

(covariance) = covariance(sys.argv[1], sys.argv[2])

# For convenience the output will be saved in one line
aux = ''
for element in covariance:
    aux = aux + str(element) + ' '
print aux

#EOF
