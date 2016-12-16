# -*- coding: utf-8 -*-

"""
This script computes the covariance of fluctuations between two vectors.
The output is a vector.

Example:  covariance_centerofmass = <mu*nu> - <mu><nu>

Author: DiegoDZ
Date: 29 june 2016
Modified: 16 december 2016

run: >> python convariance_vv.py vector1 vector2 > output_file

"""
import numpy as np
import sys

def covariance(arg1, arg2):

    # load file
    datafile1 = np.loadtxt(str(arg1))
    datafile2 = np.loadtxt(str(arg2))
    # define number snapshots
    number_snapshots = len(datafile1)

    covariance = np.sum(datafile1[:,1] * datafile2[:,1]) / number_snapshots - np.sum(datafile1[:,1]) * np.sum(datafile2[:,1]) / number_snapshots ** 2

    return covariance

(covariance) = covariance(sys.argv[1], sys.argv[2])

print covariance

#EOF
