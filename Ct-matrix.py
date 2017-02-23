#######################################################
# This scripts computes the matrix of correlations C(t)
#######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python Ct-matrix.py
#################################################################################################
#################################################################################################

##############################
#Structure of C(t)
##############################
#     | (t=1) corr_rhorho corr_rhoepsilon corr_rhog corr_epsilonrho corr_epsilonepsilon corr_epsilong corr_grho corr_gepsilon corr_gg|
#     | (t=2) corr_rhorho corr_rhoepsilon corr_rhog corr_epsilonrho corr_epsilonepsilon corr_epsilong corr_grho corr_gepsilon corr_gg|
#     |    .                                                                                                                         |
#C(t)=|    .                                                                                                                         |
#     |     .                                                                                                                        |
#     | (t=n) corr_rhorho corr_rhoepsilon corr_rhog corr_epsilonrho corr_epsilonepsilon corr_epsilong corr_grho corr_gepsilon corr_gg|
##############################
##############################

import numpy as np

# Load files
corr_rhorho = np.loadtxt('corr_rhorho')
corr_rhoepsilon = np.loadtxt('corr_rhoepsilon')
corr_rhog = np.loadtxt('corr_rhog')
corr_epsilonrho = np.loadtxt('corr_epsilonrho')
corr_epsilonepsilon = np.loadtxt('corr_epsilonepsilon')
corr_epsilong = np.loadtxt('corr_epsilong')
corr_grho = np.loadtxt('corr_grho')
corr_gepsilon = np.loadtxt('corr_gepsilon')
corr_gg = np.loadtxt('corr_gg')

# Define variables and arrays
number_correlations_files = 9
number_nodes = np.sqrt(len(corr_rhorho[0]))
number_snapshots = len(corr_rhorho)
Ct = np.zeros((number_snapshots, number_correlations_files * number_nodes ** 2))

# Concatenate the rows of the correlations files in order to create the matrix of correlations C(t)
for i in range(number_snapshots):
    Ct[i,:] = np.hstack((corr_rhorho[i,:], corr_rhoepsilon[i,:], corr_rhog[i,:], corr_epsilonrho[i,:], corr_epsilonepsilon[i,:], corr_epsilong[i,:], corr_grho[i,:], corr_gepsilon[i,:], corr_gg[i,:]))

# Save C(t) matrix
np.savetxt('Ctmatrix', Ct)

#EOF
