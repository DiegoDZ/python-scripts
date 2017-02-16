#################################################################################################
# This scripts computes the prediction of the evolution of the fluid density and the fluid energy
#################################################################################################

#Author: DiegoDZ
#Date: Feb 2017
#Run: >> python model_rhoeTheory.py
#################################################################################################

import numpy as np
from scipy.linalg import expm

# Load files
density = np.loadtxt('mesoDensity_fluid')
energy = np.loadtxt('mesoEnergy_fluid')
D = np.loadtxt('D7-matrix')
H = np.loadtxt('H-matrix')

# Select the interesting part of the profiles
mesoDensity_fluid_100 = density[100:200, :]
mesoEnergy_fluid_100 = energy[100:200, :]

# Define variables
time = 0.5
Ntime = 100
dt = 0.005
number_nodes = len(density[0])
number_snapshots = len(density)

# Concatenate the row 100 of the density file and row 100 of the energy file. Solid heated.
A_zero = np.hstack((density[100,:], energy[100,:]))
# Concatenate the last row of the density file and the last row of the energy file. Equilibrium.
A_inf = np.hstack((density[number_snapshots-1,:], energy[number_snapshots-1,:]))

A_evol = np.zeros((number_nodes * 2, Ntime))
# D, H, A_zero and A_evol as matrices
D = np.asmatrix(D)
H = np.asmatrix(H)
A_zero = np.asmatrix(A_zero)
A_inf = np.asmatrix(A_inf)
A_evol = np.asmatrix(A_evol)

# Calculate the product D7H
#DH = D.dot(H)

for i in range(0, Ntime, 1):
    A_evol[:,i] = expm(-np.dot(D, H) * i * dt ).dot((A_zero-A_inf).T) + A_inf.T

rho_evolution = A_evol[0:number_nodes, :]
e_evolution = A_evol[number_nodes + 1 : number_nodes * 2, :]

# Save files
np.savetxt('mesoDensity_fluid_100', mesoDensity_fluid_100)
np.savetxt('mesoEnergy_fluid_100', mesoEnergy_fluid_100)
np.savetxt('rho_evolution_rhoeTheory', rho_evolution.T)
np.savetxt('e_evolution_rhoeTheory', e_evolution.T)
#np.savetxt('DH-matrix', DH)

#EOF
