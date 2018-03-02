#------------------------------------------------------------------------------
#                            computeLambda-sf.py
#------------------------------------------------------------------------------
# This script computes the matrix lambda
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : January 2018
#                         Modified : January 2018
#                         Run      : python computeLambda-sf.py  (Run with python 2.7)
#------------------------------------------------------------------------------

import numpy as np
import datetime
from scipy import linalg
from scipy.linalg import expm
#from scipy.linalg import norm


#------------------------------------------------------------------------------
# Define global variables
#------------------------------------------------------------------------------
Lx,Ly,Lz    = 40.0,40.0,33.0  #dimensions of the simulation box   #REVISAR
totalNodes  = 66              #number of nodes
nNodes      = 61              #number of fluid nodes
dz          = Lz/totalNodes   #bin size
V           = dz * Lx * Ly    #bin volume
dt          = 0.004           #lammps dt=2e-3 (but info saved every 2 steps)
nSteps      = 7500            #t=30 (r.u.). The 'support' of the correlation files after cut them and the support of the C(t) predicted
tol         = 1e-3            #rcond in linalg.pinv. It will be use to compute R
tauStart    = 0.1
tauStop     = 0.4
tauDump     = 0.02
nBlocks     = 1
nVar        = int(np.sqrt(nBlocks))
dim         = nVar * nNodes

eps                                 = np.identity(dim)
eps[dim-nNodes:dim,dim-nNodes:dim] *= -1

#------------------------------------------------------------------------------
# Define functions
#------------------------------------------------------------------------------

#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(nVar,nVar,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(nVar,nNodes,nVar,nNodes).swapaxes(1,2).ravel()
    return B

#Lambda
def computeLambda(Ct,C0):
    row       = int(round(tau / dt))
    Cforward  = reshape_vm(Ct[row+1,:])
    Cbackward = reshape_vm(Ct[row-1,:])
    Cdev      = (Cforward - Cbackward) / (2 * dt)
    Cdev0     = (reshape_vm(Ct[1,:]) - reshape_vm(Ct[1,:]).T)/ (2 * dt)
    Cinv      = linalg.pinv(reshape_vm(Ct[row]), rcond = tol)
    CinvSim   = 0.5 * (Cinv + Cinv.T)
    L0        = - Cdev0
    Lanti     = 0.5 * (L0 - L0.T)                                                     #antisymmetric
    L         = 0.5 * (Lanti + eps.dot(Lanti.T).dot(eps))                             #onsager
    Lambda    = - Cdev.dot(Cinv)
    M0        = Lambda.dot(C0) - L
    Mstar         = 0.5 * (M0 + eps.dot(M0.T).dot(eps))                                   #onsager
    return Lambda, Mstar, L, Cdev0, Cdev, Cinv, CinvSim

#C(t) predicted
def computeCtpredict(Ct,Lambda):
    Ctpredict      = np.zeros((nSteps, nNodes**2*nBlocks))
    row = int(round(tau / dt))
    t   = 0
    for j in range(nSteps):
        print datetime.datetime.now(), 'Computing C(t) predicted. Step', str(j), 'tau=', str(tau)
        Ctpredict[j,:]    = reshape_mv(np.dot(expm(-Lambda * (t-tau)), reshape_vm(Ct[row])))
        t+=dt
    return Ctpredict

#-----------------------------------------------------------------------------------------------------------------------------
#                                                START COMPUTATION
#----------------------------------------------------------------------------------------------------------------------------

print datetime.datetime.now(),'Computing lambda, Cdev and Cinv...'
Ct                = np.loadtxt('Ct-sf-500steps.dat')
C0                = np.loadtxt('C-sf-t0.dat')
#LambdaMovie       = np.zeros((int(round((tstop-tstart)/tdump)),nNodes**2))
i=0
LambdaDiag = np.zeros((nNodes, int((tauStop-tauStart)/tauDump+1)))
print np.shape(LambdaDiag)
for tau in np.arange(tauStart, tauStop, tauDump):
    print i
    Lambda, Mstar, L, Cdev0, Cdev, Cinv, CinvSim = computeLambda(Ct,C0)
    Ctpredict = computeCtpredict(Ct,Lambda)
    LambdaDiag[:,i] = np.diag(Lambda)
    #Movie
    #LambdaMovie[i,:]      = reshape_mv(Lambda)
    np.savetxt('Lambda-sf-tau'+str(tau)+'.dat', Lambda)
    np.savetxt('Cdev-sf-tau'+str(tau)+'.dat', Cdev)
    np.savetxt('Cinv-sf-tau'+str(tau)+'.dat', Cinv)
    np.savetxt('Ctpredict-sf-tau'+str(tau)+'.dat', Ctpredict)
    #np.savetxt('Mstar-sf-tau'+str(tau), M)
    #np.savetxt('L-sf-tau'+str(tau), L)
    #np.savetxt('Cdev0-sf-tau'+str(tau), Cdev0)
    i+=1
np.savetxt('LambdaDiag-sf.dat', LambdaDiag)
#np.savetxt('LambdaMovie-sf.dat', LambdaMovie)
print datetime.datetime.now(),'Lambda, Cdev and Cinv computed!'



