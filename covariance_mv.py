# -*- coding: utf-8 -*-

"""
This script computes the covariance  between a matrix and a vector.
The output is a vector.

Example:  covariance_centerofmass = <mu*nu> - <mu><nu>

Author: DiegoDZ
Date: 29 june 2016

run: >> python convariance_mv.py matrix vector > output_file

"""
import numpy as np
import sys

def covariance(arg1, arg2):

    # load file
    datafile1 = np.loadtxt(str(arg1))
    datafile2 = np.loadtxt(str(arg2))
    # define number of nodes
    number_nodes = len(datafile1[0])
    # define number of snapshots
    number_snapshots = len(datafile1)
    # create an array in which we will be the output.
    covariance= np.zeros(number_nodes)

    for i in range(0,number_nodes):
        covariance[i] = np.sum(datafile1[:,i] * datafile2[:,1]) / number_snapshots - np.sum(datafile1[:,i]) * np.sum(datafile2[:,1]) / number_snapshots ** 2

    return covariance

(covariance) = covariance(sys.argv[1], sys.argv[2])

# For convenience the output will be saved in one column
aux = ''
for element in covariance:
    aux = aux + str(element) + '\n'
print aux

#EOF
