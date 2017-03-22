######################################################
# This scripts computes the matrix of covariances C(t)
######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python Ct-matrix.py
#################################################################################################
#################################################################################################

##############################
#Structure of C(t)
##############################
#     | (t=1) corr_rhorho corr_rhoe corr_rhog corr_erho corr_ee corr_eg corr_grho corr_ge corr_gg|
#     |    .                                                                                     |
#C(t)=|    .                                                                                     |
#     |    .                                                                                     |
#     | (t=n) corr_rhorho corr_rhoe corr_rhog corr_erho corr_ee corr_eg corr_grho corr_ge corr_gg|
##############################
##############################

import numpy as np

# Load files
corr_rhorho = np.loadtxt('corr_rhorho')
corr_rhoe = np.loadtxt('corr_rhoe')
corr_rhog = np.loadtxt('corr_rhog')
#corr_rhoE = np.loadtxt('corr_rhoE')
corr_erho = np.loadtxt('corr_erho')
corr_ee = np.loadtxt('corr_ee')
corr_eg = np.loadtxt('corr_eg')
#corr_epsilonE = np.loadtxt('corr_epsilonE')
corr_grho = np.loadtxt('corr_grho')
corr_ge = np.loadtxt('corr_ge')
corr_gg = np.loadtxt('corr_gg')
#corr_gE = np.loadtxt('corr_gE')
#corr_Erho = np.loadtxt('corr_Erho')
#corr_Eepsilon = np.loadtxt('corr_Eepsilon')
#corr_Eg = np.loadtxt('corr_Eg')
#corr_EE = np.loadtxt('corr_EE')

# Define variables and arrays
number_correlations_files_rhoe = 4
number_correlations_files_rhoeg = 9
number_nodes = np.sqrt(len(corr_rhorho[0]))
number_snapshots = len(corr_rhorho)
Ct_rhoe = np.zeros((number_snapshots, number_correlations_files_rhoe * number_nodes ** 2))
Ct_rhoeg = np.zeros((number_snapshots, number_correlations_files_rhoeg * number_nodes ** 2))

# Concatenate the rows of the correlations files in order to create the matrix of correlations C(t)
for i in range(number_snapshots):
    Ct_rhoe[i,:] = np.hstack((corr_rhorho[i,:], corr_rhoe[i,:], corr_erho[i,:], corr_ee[i,:]))
    Ct_rhoeg[i,:] = np.hstack((corr_rhorho[i,:], corr_rhoe[i,:], corr_rhog[i,:], corr_erho[i,:], corr_ee[i,:], corr_eg[i,:], corr_grho[i,:], corr_ge[i,:], corr_gg[i,:]))

# Save C(t) matrix
np.savetxt('Ctmatrix_rhoe', Ct_rhoe)
np.savetxt('Ctmatrix_rhoeg', Ct_rhoeg)

#EOF
