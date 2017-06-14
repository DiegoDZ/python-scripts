#################################################################################################################
# Thi scripts creates the laplace transform of C(t), taking the laplace transform of the different blocks of C(t)
#################################################################################################################
#Author: DiegoDZ
#Date: June 2017
#Run: >> python Clmatrix.py
#################################################################################################
#################################################################################################

import numpy as np

#Load files
Cl_rhorho = np.asmatrix(np.loadtxt('Cl0_rhorho'))
Cl_rhoe   = np.asmatrix(np.loadtxt('Cl0_rhoe'))
Cl_rhog   = np.asmatrix(np.loadtxt('Cl0_rhogz'))
Cl_erho   = np.asmatrix(np.loadtxt('Cl0_erho'))
Cl_ee     = np.asmatrix(np.loadtxt('Cl0_ee'))
Cl_egz    = np.asmatrix(np.loadtxt('Cl0_egz'))
Cl_gzrho  = np.asmatrix(np.loadtxt('Cl0_gzrho'))
Cl_gze    = np.asmatrix(np.loadtxt('Cl0_gze'))
Cl_gzgz   = np.asmatrix(np.loadtxt('Cl0_gzgz'))

#Create Cl
Cl = np.bmat(([Cl_rhorho,Cl_rhoe,Cl_rhog],[Cl_erho,Cl_ee,Cl_egz],[Cl_gzrho,Cl_gze,Cl_gzgz]))

#Save C(t) matrix
np.savetxt('Cl0', Cl)
#EOF
