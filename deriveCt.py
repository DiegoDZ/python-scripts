#######################################################
# This scripts derives C(t)
#######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python deriveCt.py
#################################################################################################
#################################################################################################

##############################
#Structure of C(t)
##############################
#     | (t=1) corr_rhorho corr_rhoe corr_rhog corr_erho corr_ee corr_eg corr_grho corr_ge corr_gg corr_Erho corr_Ee corr_Eg corr_EE  |
#     |    .                                                                                                                         |
#C(t)=|    .                                                                                                                         |
#     |    .                                                                                                                         |
#     | (t=n) corr_rhorho corr_rhoe corr_rhog corr_erho corr_ee corr_eg corr_grho corr_ge corr_gg  corr_Erho corr_Ee corr_Eg corr_EE |
##############################
##############################

import numpy as np

# Load the matrix of correlations, C(t)
C = np.loadtxt('Ctmatrix')

# Define variables and arrays
dt = 0.005
number_correlations_files = 9
number_snapshots = len(C)
number_nodes = int(np.sqrt(len(C[0]) / number_correlations_files ))
Cdev = np.zeros((number_snapshots, number_correlations_files * number_nodes ** 2))

# Derive C(t)
for t in range(1,number_snapshots):
    for i in range(0, len(C[0]), 1):
        Cdev[t,i] = (C[t,i] - C[t-1,i]) / dt

# Save the derivative of C(t)
np.savetxt('Ctdev', Cdev)

#EOF


## Load the matrix of correlations, C(t)
#Ct = np.loadtxt('Cprueba')
#
## Define variables and arrays
#dt = 2
#number_correlations_files = 1
#number_snapshots = len(Ct)
#number_nodes = 2
#Ctdev = np.zeros((number_snapshots, number_correlations_files* number_nodes))
#
## Derive C(t)
## Cuidado porque C(t=-1) corresponde a la ultima fila de C(t), por lo que al calcular la derivada en t=0 estamos usando C(t=0) y C(t=-1)
#for t in range(0,number_snapshots):
#    for i in range(0, number_nodes, 1):
#        Ctdev[t,i] = (Ct[t,i] - Ct[t-1,i]) / dt
#
## Save the derivative of C(t)
#np.savetxt('Ctdev', Ctdev)
#
##EOF

