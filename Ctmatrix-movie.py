###########################################################################################################################################
# This scripts computes the matrix of correlations C(t) which will be use to watch the movie (see cutCorrelations.sh and movie-matrices.py)
###########################################################################################################################################
#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python Ctmatrix-movie.py

##############################
#Structure of C(t)
##############################
#     | (t=1) c_rhorho c_rhoe c_rhog c_erho c_ee c_eg c_grho c_ge c_gg |
#     |    .                                                           |
#C(t)=|    .                                                           |
#     |    .                                                           |
#     | (t=n) c_rhorho c_rhoe c_rhog c_erho c_ee c_eg c_grho c_ge c_gg |
#################################################################################################
#################################################################################################

import numpy as np

#Load files
c_rhorho = np.loadtxt('corr_rhorho.movie')
c_rhoe   = np.loadtxt('corr_rhoe.movie')
c_rhog   = np.loadtxt('corr_rhogz.movie')
c_erho   = np.loadtxt('corr_erho.movie')
c_ee     = np.loadtxt('corr_ee.movie')
c_eg     = np.loadtxt('corr_egz.movie')
c_grho   = np.loadtxt('corr_gzrho.movie')
c_ge     = np.loadtxt('corr_gze.movie')
c_gg     = np.loadtxt('corr_gzgz.movie')

#Define variables and arrays
nBlocks = 9
sBlocks = int(np.sqrt(nBlocks))
nNodes  = int(np.sqrt(len(c_rhorho[0])))
dim     = sBlocks * nNodes
steps   = len(c_rhorho)
Cmovie       = np.zeros((steps, nBlocks * nNodes ** 2))

#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#Concatenate the rows of the correlations files in order to create the matrix of correlations C(t)
for i in range(steps):
    C1 = np.hstack((c_rhorho[i,:], c_rhoe[i,:], c_rhog[i,:], c_erho[i,:], c_ee[i,:], c_eg[i,:], c_grho[i,:], c_ge[i,:], c_gg[i,:]))
    C2 = np.hstack((c_rhorho[i,:], c_rhoe[i,:], -c_rhog[i,:], c_erho[i,:], c_ee[i,:], -c_eg[i,:], -c_grho[i,:], -c_ge[i,:], c_gg[i,:]))
    Cmovie[i,:] = reshape_mv((reshape_vm(C1) + reshape_vm(C2).T) / 2) #increase statistic because of time reversal property

#Save C(t) matrix
np.savetxt('Cmovie', Cmovie)
#EOF
