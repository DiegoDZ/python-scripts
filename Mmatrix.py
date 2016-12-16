# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:36:25 2016

Author: DiegoDZ

Compute matrix M

"""

import numpy as np

#Load time correlation files
C_QQ = np.loadtxt('corr_QQ')
C_QPi = np.loadtxt('corr_QPi')
C_PiQ = np.loadtxt('corr_PiQ')
C_PiPi = np.loadtxt('corr_PiPi')
C_iLeiLe = np.loadtxt('corr_iLeiLe')

#Define variables
#Box length
Lx = 17.3162
Ly = 17.3162
Lz = 34.6325
#Number nodes 
number_nodes = 100
#Number nodes fluid
number_nodes_fluid = np.sqrt(len(C_QQ[0]))
#Number bines
number_bines = number_nodes - 1
#Bin size (z axis)
binSize = Lz / number_bines
#Bin volume
V = binSize * Lx * Ly
#Time step
dt = 0.005

#Compute K, L, N and S matrix
k = np.sum(C_QQ, axis = 0)
K = k.reshape(number_nodes_fluid, number_nodes_fluid) * dt
l = np.sum(C_QPi, axis = 0)
L = l.reshape(number_nodes_fluid, number_nodes_fluid) * dt
n = np.sum(C_PiQ, axis = 0)
N = n.reshape(number_nodes_fluid, number_nodes_fluid) * dt
s = np.sum(C_PiPi, axis = 0)
S = s.reshape(number_nodes_fluid, number_nodes_fluid) * dt

#Create matrix F
F = np.zeros((number_nodes_fluid, number_nodes_fluid))
F -= np.eye(number_nodes_fluid, k=1)
#F -= np.diag(np.ones(number_nodes_fluid-1), k = 1)
np.fill_diagonal(F,1)
F /= binSize

#Compute matrix M from F, K, L, H and S. And from iLepsilon
#M = V * (-F.T * K * F + L * F - F.T * N + S)
M = V * (reduce(np.dot, [F.T, K, F]) - np.dot(F.T,L) - np.dot(N, F) + S)
e = np.sum(C_iLeiLe, axis = 0)
M_e  = e.reshape(number_nodes_fluid, number_nodes_fluid) * dt

#Save result
np.savetxt('M-matrix', M)
np.savetxt('Me-matrix', M_e)
np.savetxt('K-matrix', K)
np.savetxt('L-matrix', L)
np.savetxt('N-matrix', N)
np.savetxt('S-matrix', S)

#EOF
