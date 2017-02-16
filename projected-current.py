# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:02:58 2016

This script computes the projected current of the internal energy of the fluid, the heat fluxed Q and Pi, the time derivative of the internal energy of the fluid (iLe).

Author: DiegoDZ
run: >> python projected-current.py

"""

import numpy as np

#Load files
e = np.loadtxt('InternalEnergy')
Q = np.loadtxt('Q')
Pi = np.loadtxt('Pi')
iLe = np.loadtxt('iLe')
cov_ee = np.loadtxt('cov_ee')
cov_eQ = np.loadtxt('cov_eQ')
cov_ePi = np.loadtxt('cov_ePi')
cov_eiLe = np.loadtxt('cov_eiLe')

#Define variables
number_nodes = len(e[0])
number_snapshots = len(e)
projCurr_Q = np.zeros((number_snapshots, number_nodes))
projCurr_Pi = np.zeros((number_snapshots, number_nodes))
projCurr_iLe = np.zeros((number_snapshots, number_nodes))

#Transform the array cov_ee into matrix in order to compute its inverse.
#Redefine covariances.
#It will be easier to work with the new ones.
cov_ee = np.asmatrix(cov_ee)
cov_ee_inv = np.asarray(cov_ee.I)
cov_eiLe = np.dot(cov_ee_inv, cov_eiLe)
cov_eQ = np.dot(cov_ee_inv, cov_eQ)
cov_ePi = np.dot(cov_ee_inv, cov_ePi)

#Compute projected currents
e_avg = np.sum(e, axis = 0)/ number_snapshots
E = e - e_avg

for i in range (0, number_snapshots, 1):
    for j in range (0, number_nodes, 1):
        projCurr_Q[i,j] = Q[i,j] - np.sum(E[i,:] * cov_eQ[:,j])
        projCurr_Pi[i,j] = Pi[i,j] - np.sum(E[i,:] * cov_ePi[:,j])
        projCurr_iLe[i,j] = iLe[i,j] - np.sum(E[i,:] * cov_eiLe[:,j])

#Save output
np.savetxt('projCurr_Q', projCurr_Q)
np.savetxt('projCurr_Pi', projCurr_Pi)
np.savetxt('projCurr_iLe', projCurr_iLe)
