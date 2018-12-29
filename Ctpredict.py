#########################################################################
#THIS SCRIPT COMPUTES C(t) PREDICTED AND ITS ERROR FROM FIRST DERIVATIVES
#########################################################################
#Author: DiegoDZ
#Date:   jun 2017
#########################################################################
import numpy as np
from scipy.linalg import expm
from scipy.linalg import norm
import datetime

#####################################
#LOAD FILES AND DEFINE VARIBLES
#####################################
print datetime.datetime.now(), 'Loading files...'
C0           = np.loadtxt('C0stat')
Cshort       = np.loadtxt('Ctstat_2e3steps')
nRows, nCols = np.shape(C0)
nBlocks      = 9
sBlocks      = int(np.sqrt(nBlocks))
nNodes       = len(C0[0]) / int(np.sqrt(nBlocks))
dim          = sBlocks * nNodes
dt           = 0.002 #dt=0.0002 lammps. Saved output each 10 nSteps -> 0.002 between snapshots
nSteps       = 2000

#####################################
#DEFINE FUNCTIONS
#####################################
#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B
#Change format: Matrix-> Vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B
#Error
def frobenious(A,B):
    error = norm((reshape_vm(A - B)), 'fro') / nNodes
    return error

#####################################
#COMPUTE C(t) PREDICTED AND ITS ERROR
#####################################
tauStart = 0.03
tauStop  = 0.05
tauJump  = 0.01
for i in np.arange(tauStart, tauStop, tauJump):
    Lambda         = np.loadtxt('Lambda'+str(i))
    Ctpredict      = np.zeros((nSteps, nNodes**2*nBlocks))
    errorCtpredict = np.zeros(nSteps)
    t=0
    for j in range(nSteps):
        print datetime.datetime.now(), 'Computing step', str(j), 'for lambda', str(i)
        Ctpredict[j,:]    = reshape_mv(np.dot(expm(-Lambda * t),C0))
        errorCtpredict[j] = frobenious(Ctpredict[j,:],Cshort[j,:])
        t+=dt
    print datetime.datetime.now(), 'Saving output for lambda', str(i)
    np.savetxt('Ctpredict' + str(i), Ctpredict)
    np.savetxt('errorCtpredict' + str(i), errorCtpredict)
#EOF
