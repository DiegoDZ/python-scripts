#------------------------------------------------------------------------------
#                                   pbc.py
#------------------------------------------------------------------------------
# This script takes a file of correlations as an input and computes a new one
# in the Fourier space. The ouput is the new file of correlations, its real and
# imaginary part and the diagonal of the real part (which corresponds to its
# eigenvalues)
#------------------------------------------------------------------------------
#                         Author: @DiegoDZ
#                         Date  : September 2017
#                         Run   : python pbc.py arg  (Run with python 2.7)
#                         arg   : rhorho, rhoe, rhogx, rhogz, erho, ee, egx, egz,
#                                 gxrho, gxe, gxgz, gzrho, gze, gzgx, gzgz
#------------------------------------------------------------------------------

import numpy as np
import sys

#------------------------------------------------------------------------------
# Define variables
#------------------------------------------------------------------------------
nSteps    = 2000                    #t=4 (r.u.).
nNodes    = 55                      #number of nodes
nBlocks   = 1                       #number of blocks. In this case is 1 because
                                    #it is used only one file of correaltions.
sBlocks   = int(np.sqrt(nBlocks))
dim       = sBlocks * nNodes
variables = sys.argv[1]

#------------------------------------------------------------------------------
# Define functions
#------------------------------------------------------------------------------
#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#--------------------------------------------------------------------------------------------------------------------------------------
#                                                START COMPUTATION
#--------------------------------------------------------------------------------------------------------------------------------------
c                    = np.loadtxt('corr_'+str(variables)+'_2e3steps')
E                    = np.zeros((nNodes, nNodes), dtype = complex)
Einv                 = np.zeros((nNodes, nNodes), dtype = complex)
Cfourier             = np.zeros((nSteps, nNodes**2), dtype = complex)
CfourierReal         = np.zeros((nSteps, nNodes**2))
CfourierImag         = np.zeros((nSteps, nNodes**2))
CfourierRealDiag     = np.zeros((nSteps, nNodes))
CfourierRealDiagNorm = np.zeros((nSteps, nNodes))

for mu in range(nNodes):
    for nu in range(nNodes):
            E[mu,nu]    = np.exp( 1j*2*np.pi*mu*nu/nNodes)/np.sqrt(nNodes)
            Einv[mu,nu] = np.exp(-1j*2*np.pi*mu*nu/nNodes)/np.sqrt(nNodes)

Cfourier0         = np.dot(Einv, reshape_vm(c[0,:])).dot(E)
CfourierReal0     = Cfourier0.real
CfourierRealDiag0 = np.diag(CfourierReal0)

for i in range(nSteps):
    Cfourier[i,:]             = reshape_mv(np.dot(Einv, reshape_vm(c[i,:])).dot(E))
    CfourierReal[i,:]         = Cfourier[i,:].real
    CfourierRealDiag[i,:]     = np.diag(reshape_vm(CfourierReal[i,:]))
    CfourierRealDiagNorm[i,:] = np.diag(reshape_vm(CfourierReal[i,:])) / CfourierRealDiag0
    CfourierImag[i,:]         = Cfourier[i,:].imag

#np.savetxt('Cfourier-'+str(variables), Cfourier)
#np.savetxt('CfourierReal-'+str(variables), CfourierReal)
#np.savetxt('CfourierImag-'+str(variables), CfourierImag)
np.savetxt('CfourierRealDiag-'+str(variables), CfourierRealDiag)
np.savetxt('CfourierRealDiagNorm-'+str(variables), CfourierRealDiagNorm)
#--------------------------------------------------------------------------------------------------------------------------------------
#                                                END COMPUTATION
#--------------------------------------------------------------------------------------------------------------------------------------
