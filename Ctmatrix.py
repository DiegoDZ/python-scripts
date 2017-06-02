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

# Load files
c_rhorho = np.loadtxt('c_rhorho.avgs.dat')
c_rhoe   = np.loadtxt('c_rhoe.avgs.dat')
c_rhog   = np.loadtxt('c_rhogz.avgs.dat')
c_erho   = np.loadtxt('c_erho.avgs.dat')
c_ee     = np.loadtxt('c_ee.avgs.dat')
c_eg     = np.loadtxt('c_egz.avgs.dat')
c_grho   = np.loadtxt('c_gzrho.avgs.dat')
c_ge     = np.loadtxt('c_gze.avgs.dat')
c_gg     = np.loadtxt('c_gzgz.avgs.dat')

# Define variables and arrays
blocks_rhoe = 4
blocks_rhoeg = 9
nodes = np.sqrt(len(c_rhorho[0]))
steps = len(c_rhorho)
Ct_rhoe = np.zeros((steps, blocks_rhoe * nodes ** 2))
Ct_rhoeg = np.zeros((steps, blocks_rhoeg * nodes ** 2))

# Concatenate the rows of the correlations files in order to create the matrix of correlations C(t)
for i in range(steps):
    Ct_rhoe[i,:] = np.hstack((c_rhorho[i,:], c_rhoe[i,:], c_erho[i,:], c_ee[i,:]))
    Ct_rhoeg[i,:] = np.hstack((c_rhorho[i,:], c_rhoe[i,:], c_rhog[i,:], c_erho[i,:], c_ee[i,:], c_eg[i,:], c_grho[i,:], c_ge[i,:], c_gg[i,:]))

# Save C(t) matrix
np.savetxt('Ctmatrix_rhoe', Ct_rhoe)
np.savetxt('Ctmatrix_rhoeg', Ct_rhoeg)

#EOF
