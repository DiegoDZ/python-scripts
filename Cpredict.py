##################################################
#THIS SCRIPT COMPUTES C(t) PREDICTED AND ITS ERROR
##################################################
#AUTHOR: DiegoDZ
#DATE: jun 2017
#RUN: >>> Cpredict.py
##################################################
##################################################
import numpy as np
from scipy.linalg import expm

#Load files and define variables
C0            = np.loadtxt('C0')
Cshort        = np.loadtxt('Cshort_1e3steps')
Lambda        = np.loadtxt('LambdaSimRec')
nRows, nCols  = np.shape(C0)
nBlocks       = 9
sBlocks       = int(np.sqrt(nBlocks))
nNodes        = len(C0[0]) / int(np.sqrt(nBlocks))
dt            = 0.002 #dt=0.0002 lammps. Saved output each 10 steps -> 0.002 between snapshots
steps         = 1000

#Change format: Matrix-> Vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#C(t) predicted and its error
Cpredict = np.zeros((steps, nNodes**2*nBlocks))
t=0
for i in range(steps):
    #Cpredict[i,:] = reshape_mv(np.dot(expm(-np.dot(C0,Cl0inv) * t),C0))
    Cpredict[i,:] = reshape_mv(np.dot(expm(-Lambda * t),C0))
    t+=dt

errorCpredict = Cpredict - Cshort

#Save output
np.savetxt('Cpredict', Cpredict)
np.savetxt('errorCpredict', errorCpredict)
#EOF
