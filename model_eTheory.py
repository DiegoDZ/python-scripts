###########################################################################
# This scripts computes the prediction of the evolution of the fluid energy
###########################################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python model_eTheory.py
###########################################################################

import numpy as np
from scipy.linalg import expm

#Load files
energy = np.loadtxt('mesoEnergy_fluid')
He = np.loadtxt('He-matrix')
De = np.loadtxt('Dee7')

#Select the interesting part of the profiles
mesoEnergy_fluid_100 = energy[100:200, :]

# Define variables
time = 0.5
Ntime = 100
dt = 0.005
number_nodes = len(energy[0])
number_snapshots = len(energy)

#Select the row 100 (solid heated) and the last row of the energy file (equilibrium)
A_zero = energy[100,:]
A_inf = energy[number_snapshots-1,:]

A_evol = np.zeros((number_nodes, Ntime))
# D, H, A_zero and A_evol as matrices
De = np.asmatrix(De)
He = np.asmatrix(He)
A_zero = np.asmatrix(A_zero)
A_inf = np.asmatrix(A_inf)
A_evol = np.asmatrix(A_evol)

for i in range(0, Ntime, 1):
    A_evol[:,i] = expm(-np.dot(De, He) * i * dt ).dot((A_zero-A_inf).T) + A_inf.T

np.savetxt('e_evolution_eTheory', A_evol.T)

#EOF


