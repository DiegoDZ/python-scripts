#####################
# Running-integral.py
#####################

# This script calculates the cumulative integral of a function using the sub-package scipy.integrate.
# Caution: the script is used for calculate the cumulative integral of the specified column of a matrix input. 
# More info: https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/integrate.html

# Author: DiegoDZ
# Date: 15 december 2016
# Run: python running-integral.py input > output


import scipy.integrate as integrate
import numpy as np
import sys


def runningIntegral(arg1):

    column = 1

    A = np.loadtxt(str(arg1))
    B = integrate.cumtrapz(A[:,column],x=None, dx = 0.005, initial=0)

    return B

B = runningIntegral(sys.argv[1])

aux = ''
for element in B:
    aux = aux + str(element) + '\n'
print aux
