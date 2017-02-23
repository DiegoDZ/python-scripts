#######################################################
# This scripts derives C(t)
#######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python deriveCt-matrix.py
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

# Load the matrix of correlations, C(t)
Ct = np.loadtxt('Ctmatrix')

# Define variables and arrays
dt = 0.005
number_correlations_files = 9
number_snapshots = len(Ct)
number_nodes = int(np.sqrt(len(Ct[0]) / number_correlations_files ))
Ctdev = np.zeros((number_snapshots, number_correlations_files * number_nodes ** 2))

# Derive C(t)
# Cuidado porque C(t=-1) corresponde a la ultima fila de C(t), por lo que al calcular la derivada en t=0 estamos usando C(t=0) y C(t=-1)
for t in range(0,number_snapshots):
    for i in range(0, number_nodes, 1):
        Ctdev[t,i] = (Ct[t,i] - Ct[t-1,i]) / dt

# Save the derivative of C(t)
np.savetxt('Ctdev', Ctdev)

#EOF
