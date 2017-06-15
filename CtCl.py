######################################################
# This scripts computes the matrix of correlations C(t)
######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python Ctmatrix.py
#################################################################################################
#################################################################################################

##############################
#Structure of C(t)
##############################
#     | (t=1) c_rhorho c_rhoe c_rhog c_erho c_ee c_eg c_grho c_ge c_gg |
#     |    .                                                           |
#C(t)=|    .                                                           |
#     |    .                                                           |
#     | (t=n) c_rhorho c_rhoe c_rhog c_erho c_ee c_eg c_grho c_ge c_gg |
##############################
##############################

import numpy as np
import datetime

print datetime.datetime.now(), 'Loading files'
# Load files
c_rhorho = np.loadtxt('corr_rhorho.avgs.dat.SHORT')
c_rhoe   = np.loadtxt('corr_rhoe.avgs.dat.SHORT')
c_rhog   = np.loadtxt('corr_rhogz.avgs.dat.SHORT')
c_erho   = np.loadtxt('corr_erho.avgs.dat.SHORT')
c_ee     = np.loadtxt('corr_ee.avgs.dat.SHORT')
c_eg     = np.loadtxt('corr_egz.avgs.dat.SHORT')
c_grho   = np.loadtxt('corr_gzrho.avgs.dat.SHORT')
c_ge     = np.loadtxt('corr_gze.avgs.dat.SHORT')
c_gg     = np.loadtxt('corr_gzgz.avgs.dat.SHORT')

# Define variables and arrays
nBlocks = 9
sBlocks = int(np.sqrt(nBlocks))
nNodes = int(np.sqrt(len(c_rhorho[0])))
dim = sBlocks * nNodes
steps = len(c_rhorho)
Ct_rhoeg = np.zeros((steps, nBlocks * nNodes ** 2))

#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

print datetime.datetime.now(), 'Computing C(t)'
# Concatenate the rows of the correlations files in order to create the matrix of correlations C(t)
for i in range(steps):
    C1 = np.hstack((c_rhorho[i,:], c_rhoe[i,:], c_rhog[i,:], c_erho[i,:], c_ee[i,:], c_eg[i,:], c_grho[i,:], c_ge[i,:], c_gg[i,:]))
    C2 = np.hstack((c_rhorho[i,:], c_rhoe[i,:], -c_rhog[i,:], c_erho[i,:], c_ee[i,:], -c_eg[i,:], -c_grho[i,:], -c_ge[i,:], c_gg[i,:]))
    Ct_rhoeg[i,:] = reshape_mv((reshape_vm(C1) + reshape_vm(C2).T) / 2) #increase statistic because of time reversal property

########
#LAPLACE
########
nRows, nCols = np.shape(Ct_rhoeg)
dt = 0.002 #dt=0.0002 lammps. Saved output each 10 steps -> 0.002 between snapshots
time = np.array(np.arange(0, nRows*dt, dt))

#C0
C0=reshape_vm(Ct_rhoeg[0,:])
C0_stat = (C0 + C0.T) / 2 #C(t=0) is symmetric

print datetime.datetime.now(), 'Computing Laplace Transform'
#laplace for s=0
s = 0
Cl = reshape_vm(np.trapz((np.exp(-s * time[:, np.newaxis]) * Ct_rhoeg), dx=dt, axis = 0))
print datetime.datetime.now()

print datetime.datetime.now(), 'Saving output'
#Save output
np.savetxt('Cl0', Cl)
np.savetxt('C0', C0_stat)
np.savetxt('Ctmatrix_rhoeg', Ct_rhoeg)
print datetime.datetime.now(), 'Job done!'
#EOF
