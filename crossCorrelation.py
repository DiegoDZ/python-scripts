# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 2016

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
    nNodes = len(A[0])

    #Cross-correlation works as follows:
    #1. Take FFT of each column of both input matrix
    #2. Multiply one resulting transform by the complex conjugate of the other
    #3. Calculate the inverse transform of the product

    a = np.zeros((nSteps, nNodes),dtype=complex)
    b = np.zeros((nSteps, nNodes),dtype=complex)
    for i in range(0,nNodes,1):
        a[:,i] = np.conjugate(np.fft.fft(A[:,i] - np.mean(A[:,i])))
        b[:,i] = np.fft.fft(B[:,i]- np.mean(B[:,i]))
    #Multiply all columns of matrix a by all columns of matrix b.
    C = (a[...,None]*b[:,None]).reshape(a.shape[0],-1)
    #Calculate the inverse
    CAB = np.zeros((nSteps, nNodes*nNodes))
    for j in range (0, nNodes*nNodes, 1):
        CAB[:,j] = np.fft.ifft(C[:,j])
    CAB /= nSteps

    return CAB[0 : nSteps/2, :]

CAB = crossCorrelation(sys.argv[1],sys.argv[2])

#Print the result in the correct format
print '\n'.join(' '.join(str(cell) for cell in row) for row in CAB)

#EOF

