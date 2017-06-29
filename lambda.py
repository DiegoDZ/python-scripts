##############################################
# THIS SCRIPT COMPUTES THE DISSIPATIVE MATRIX M(t)
#
# AUTHOR: @DiegoDZ
# DATE: MARCH 2017
#
# run: >>> python Mmatrix_rhoegTheory.py
##############################################

import numpy as np
from scipy import linalg
import datetime
##############################################
#LOAD FILES
##############################################
print datetime.datetime.now(), 'Loading files...'
Ct = np.loadtxt('Ctstat_100steps')
##############################################
#DEFINE VARIABLES
##############################################
nBlocks        = 9
tol            = 1e-3 #rcond in linalg.pinv
nNodes         = int(np.sqrt(len(Ct[0]) / nBlocks ))
sBlocks        = int(np.sqrt(nBlocks))
dim            = sBlocks * nNodes
nSteps         = len(Ct)
Lambdat        = np.zeros((nSteps-1, nNodes ** 2 * nBlocks))
Ctdev          = np.zeros((nSteps-1, nNodes ** 2 * nBlocks))
#########Simulation details########
Lx = 46.1766
Ly = 46.1766
Lz = 23.0883
dz = Lz / nNodes
V  = dz * Lx * Ly
dt = 0.002

##############################################
#DEFINE FUNCTIONS
##############################################
#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B
#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

##############################################
#COMPUTE L as -Cdev(t=0) and R as C(t=0)^-1
##############################################
#L
Ct1       = reshape_vm(Ct[1,:])
L         = - (Ct1 - Ct1.T) / (2 * dt)
L_antiSym = (L - L.T) / 2 #L is antisymmetric
#R
Ct0       = reshape_vm(Ct[0,:])
Ct0_stat  = (Ct0 + Ct0.T) / 2 #Increase the statistic because C(t=0) is symmetric
R         = linalg.pinv(Ct0_stat, rcond = tol)
Ctinv0    = R

##############################################
#START COMPUTATION
##############################################
tstart = 5
tstop  = 31
tjump  = 5
for t in range(tstart, tstop, tjump):
    print datetime.datetime.now(), 'step', str(t)
    #####################################
    # COMPUTE Cinv
    #####################################
    #Create the matrix C(t) and its inverse
    C    = reshape_vm(Ct[t,:])
    Cinv = linalg.pinv(C, rcond = tol)
    np.savetxt('Cinv'+str(t), Cinv)

    #####################################
    # DERIVE C(t)
    #####################################
    Cforward   = reshape_vm(Ct[t+1,:])
    Cbackward  = reshape_vm(Ct[t-1,:])
    Cdev       = (Cforward - Cbackward) / (2 * dt)
    #np.savetxt('Ctdev' + str(t*dt), Cdev)

    #####################################
    # COMPUTE lambda(t)
    #####################################
    #Lambda = - Cdev.dot(Cinv)
    Lambda = - Cdev.dot(R)
    np.savetxt('Lambda' + str(t*dt), Lambda)

#Compute L*R
LR  = L.dot(R)

#Save outputs
np.savetxt('R', R)
np.savetxt('L', L_antiSym)
np.savetxt('LR', LR)
print datetime.datetime.now(), 'Job done!'
#EOF
