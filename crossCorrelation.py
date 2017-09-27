# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2016
Modified on Wed Sep 27 2017

Autor: DiegoDZ

Compute the cross correlation between all columns of two input matrix, A and B,
using the Fast Fourier Transform.

run: python crossCorrelation.py input_A input_B > output

"""
import numpy as np
import sys

def crossCorrelation(arg1,arg2):

    #Load files
    A = np.loadtxt(str(arg1))
    B = np.loadtxt(str(arg2))

    #Define params
    nSteps = len(A)

    #Cross-correlation works as follows:
    #1. Take FFT of each column of both input matrix
    #2. Multiply one resulting transform by the complex conjugate of the other
    #3. Calculate the inverse transform of the product

    a = np.fft.fft(A - np.mean(A,axis=0),axis=0)
    b = np.conjugate(np.fft.fft(B - np.mean(B,axis=0),axis=0))
    #Multiply all columns of matrix a by all columns of matrix b.
    C = (a[...,None]*b[:,None]).reshape(a.shape[0],-1)
    #Calculate the inverse
    CAB = np.fft.ifft(C,axis=0).real / nSteps

    return CAB[0:nSteps/2,:]

CAB = crossCorrelation(sys.argv[1],sys.argv[2])

#Print the result in the correct format
print '\n'.join(' '.join(str(cell) for cell in row) for row in CAB)

#EOF

