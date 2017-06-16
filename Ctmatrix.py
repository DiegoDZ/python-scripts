#####################################################
#THIS SCRIPT COMPUTES THE MATRIX OF CORRELATIONS C(t)
#####################################################
#AUTHOR: DiegoDZ
#DATE: Feb 2017
#RUN: >> python Ctmatrix-movie.py

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

#Load files (see cutCorrelations.sh before you select the files)

c_rhorho = np.loadtxt('corr_rhorho.avgs.dat.SHORT_1e4steps')
c_rhoe   = np.loadtxt('corr_rhoe.avgs.dat.SHORT_1e4steps')
c_rhog   = np.loadtxt('corr_rhogz.avgs.dat.SHORT_1e4steps')
c_erho   = np.loadtxt('corr_erho.avgs.dat.SHORT_1e4steps')
c_ee     = np.loadtxt('corr_ee.avgs.dat.SHORT_1e4steps')
c_eg     = np.loadtxt('corr_egz.avgs.dat.SHORT_1e4steps')
c_grho   = np.loadtxt('corr_gzrho.avgs.dat.SHORT_1e4steps')
c_ge     = np.loadtxt('corr_gze.avgs.dat.SHORT_1e4steps')
c_gg     = np.loadtxt('corr_gzgz.avgs.dat.SHORT_1e4steps')

#c_rhorho = np.loadtxt('corr_rhorho.avgs.dat.SHORT_1e3steps')
#c_rhoe   = np.loadtxt('corr_rhoe.avgs.dat.SHORT_1e3steps')
#c_rhog   = np.loadtxt('corr_rhogz.avgs.dat.SHORT_1e3steps')
#c_erho   = np.loadtxt('corr_erho.avgs.dat.SHORT_1e3steps')
#c_ee     = np.loadtxt('corr_ee.avgs.dat.SHORT_1e3steps')
#c_eg     = np.loadtxt('corr_egz.avgs.dat.SHORT_1e3steps')
#c_grho   = np.loadtxt('corr_gzrho.avgs.dat.SHORT_1e3steps')
#c_ge     = np.loadtxt('corr_gze.avgs.dat.SHORT_1e3steps')
#c_gg     = np.loadtxt('corr_gzgz.avgs.dat.SHORT_1e3steps')

#c_rhorho = np.loadtxt('corr_rhorho.avgs.dat.MOVIE')
#c_rhoe   = np.loadtxt('corr_rhoe.avgs.dat.MOVIE')
#c_rhog   = np.loadtxt('corr_rhogz.avgs.dat.MOVIE')
#c_erho   = np.loadtxt('corr_erho.avgs.dat.MOVIE')
#c_ee     = np.loadtxt('corr_ee.avgs.dat.MOVIE')
#c_eg     = np.loadtxt('corr_egz.avgs.dat.MOVIE')
#c_grho   = np.loadtxt('corr_gzrho.avgs.dat.MOVIE')
#c_ge     = np.loadtxt('corr_gze.avgs.dat.MOVIE')
#c_gg     = np.loadtxt('corr_gzgz.avgs.dat.MOVIE')

#Define variables and arrays
nBlocks = 9
sBlocks = int(np.sqrt(nBlocks))
nNodes  = int(np.sqrt(len(c_rhorho[0])))
dim     = sBlocks * nNodes
steps   = len(c_rhorho)
C       = np.zeros((steps, nBlocks * nNodes ** 2))

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
    C[i,:] = reshape_mv((reshape_vm(C1) + reshape_vm(C2).T) / 2) #increase statistic because of time reversal property

#Save C(t) matrix
#np.savetxt('Cshort_1e4steps', C)
#np.savetxt('Cshort_1e3steps', C)
#np.savetxt('Cmovie', C)
#EOF
