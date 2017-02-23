#######################################################
# This scripts chops the matrix of correlations C(t)
#######################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python chopCt.py
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

# Load C(t)
Ct = np.loadtxt('Ctmatrix')

# Select the column we are interested in
column = raw_input("Select the correlation you are interested in: ")  #in python 2
#column = input("Select the correlation you are interested in: ")  #in python 3
Ctselection = Ct[:,column]

#Save the selection
np.savetxt('Ct-selection', Ctselection)

#EOF
