import numpy as np
from scipy.linalg import expm

#################################################################################################################
#LOAD FILE, DEFINE VARIABLES AND FUNCTIONS
#################################################################################################################
C0  = np.loadtxt('C0')
#Cl0inv = np.loadtxt('Cl0inv')
Lambda = np.loadtxt('LambdaSimRec14jun')
nRows, nCols  = np.shape(C0)
nBlocks = 9
sBlocks = int(np.sqrt(nBlocks))
nNodes = len(C0[0]) / int(np.sqrt(nBlocks))
dt = 0.002 #dt=0.0002 lammps. Saved output each 10 steps -> 0.002 between snapshots
steps = 1000

def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

##############################################
#C(t)
##############################################
Cpredict = np.zeros((steps, nNodes**2*nBlocks))
t=0
for i in range(steps):
    #Cpredict[i,:] = reshape_mv(np.dot(expm(-np.dot(C0,Cl0inv) * t),C0))
    Cpredict[i,:] = reshape_mv(np.dot(expm(-Lambda * t),C0))
    t+=dt
#Save output
np.savetxt('Cpredict14jun', Cpredict)
#EOF
