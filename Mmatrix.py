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

#Select the range of snapshots will be taken into account.
C_QQ7 = C_QQ[0:7, :]
C_QQ15 = C_QQ[0:15, :]
C_QQ50 = C_QQ[0:50, :]
C_QQ100 = C_QQ[0:100, :]
C_QPi7 = C_QPi[0:7, :]
C_QPi15 = C_QPi[0:15, :]
C_QPi50 = C_QPi[0:50, :]
C_QPi100 = C_QPi[0:100, :]
C_PiQ7 = C_PiQ[0:7, :]
C_PiQ15 = C_PiQ[0:15, :]
C_PiQ50 = C_PiQ[0:50, :]
C_PiQ100 = C_PiQ[0:100, :]
C_PiPi7 = C_PiPi[0:7, :]
C_PiPi15 = C_PiPi[0:15, :]
C_PiPi50 = C_PiPi[0:50, :]
C_PiPi100 = C_PiPi[0:100, :]
C_iLeiLe7 = C_iLeiLe[0:7, :]
C_iLeiLe15 = C_iLeiLe[0:15, :]
C_iLeiLe50 = C_iLeiLe[0:50, :]
C_iLeiLe100 = C_iLeiLe[0:100, :]

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
#number_bines = number_nodes - 1
#Bin size (z axis)
#binSize = Lz / number_bines
dz = Lz / number_nodes
#Bin volume
V = dz * Lx * Ly
#Time step
dt = 0.005

#Compute K, L, N and S matrix
k7 = np.sum(C_QQ7, axis = 0)
K7 = k7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
l7 = np.sum(C_QPi7, axis = 0)
L7 = l7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
n7 = np.sum(C_PiQ7, axis = 0)
N7 = n7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
s7 = np.sum(C_PiPi7, axis = 0)
S7 = s7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
k15 = np.sum(C_QQ15, axis = 0)
K15 = k15.reshape(number_nodes_fluid, number_nodes_fluid) * dt
l15 = np.sum(C_QPi15, axis = 0)
L15 = l15.reshape(number_nodes_fluid, number_nodes_fluid) * dt
n15 = np.sum(C_PiQ15, axis = 0)
N15 = n15.reshape(number_nodes_fluid, number_nodes_fluid) * dt
s15 = np.sum(C_PiPi15, axis = 0)
S15 = s15.reshape(number_nodes_fluid, number_nodes_fluid) * dt
k50 = np.sum(C_QQ50, axis = 0)
K50 = k50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
l50 = np.sum(C_QPi50, axis = 0)
L50 = l50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
n50 = np.sum(C_PiQ50, axis = 0)
N50 = n50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
s50 = np.sum(C_PiPi50, axis = 0)
S50 = s50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
k100 = np.sum(C_QQ100, axis = 0)
K100 = k100.reshape(number_nodes_fluid, number_nodes_fluid) * dt
l100 = np.sum(C_QPi100, axis = 0)
L100 = l100.reshape(number_nodes_fluid, number_nodes_fluid) * dt
n100 = np.sum(C_PiQ100, axis = 0)
N100 = n100.reshape(number_nodes_fluid, number_nodes_fluid) * dt
s100 = np.sum(C_PiPi100, axis = 0)
S100 = s100.reshape(number_nodes_fluid, number_nodes_fluid) * dt

#Create matrix F
F = np.zeros((number_nodes_fluid, number_nodes_fluid))
F -= np.eye(number_nodes_fluid, k=1)
#F -= np.diag(np.ones(number_nodes_fluid-1), k = 1)
np.fill_diagonal(F,1)
F /= dz

#Compute matrix M from F, K, L, H and S.
#M = V * (-F.T * K * F + L * F - F.T * N + S)
M7 = V * (reduce(np.dot, [F.T, K7, F]) - np.dot(F.T,L7) - np.dot(N7, F) + S7)
M15 = V * (reduce(np.dot, [F.T, K15, F]) - np.dot(F.T,L15) - np.dot(N15, F) + S15)
M50 = V * (reduce(np.dot, [F.T, K50, F]) - np.dot(F.T,L50) - np.dot(N50, F) + S50)
M100 = V * (reduce(np.dot, [F.T, K100, F]) - np.dot(F.T,L100) - np.dot(N100, F) + S100)
# Compute matrix M form iLepsilon
e7 = V * (np.sum(C_iLeiLe7, axis = 0))
Me7  = e7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
e15 =  V * (np.sum(C_iLeiLe15, axis = 0))
Me15  = e15.reshape(number_nodes_fluid, number_nodes_fluid) * dt
e50 =  V * (np.sum(C_iLeiLe50, axis = 0))
Me50  = e50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
e100 = V * (np.sum(C_iLeiLe100, axis = 0))
Me100  = e100.reshape(number_nodes_fluid, number_nodes_fluid) * dt

#Save results
np.savetxt('L7-matrix', L7)
np.savetxt('L15-matrix', L15)
np.savetxt('L50-matrix', L50)
np.savetxt('L100-matrix', L100)
np.savetxt('K7-matrix', K7)
np.savetxt('K15-matrix', K15)
np.savetxt('K50-matrix', K50)
np.savetxt('K100-matrix', K100)
np.savetxt('N7-matrix', N7)
np.savetxt('N15-matrix', N15)
np.savetxt('N50-matrix', N50)
np.savetxt('N100-matrix', N100)
np.savetxt('S7-matrix', S7)
np.savetxt('S15-matrix', S15)
np.savetxt('S50-matrix', S50)
np.savetxt('S100-matrix', S100)
np.savetxt('M7-matrix', M7)
np.savetxt('M15-matrix', M15)
np.savetxt('M50-matrix', M50)
np.savetxt('M100-matrix', M100)

np.savetxt('Me7-matrix', Me7)
np.savetxt('Me15-matrix', Me15)
np.savetxt('Me50-matrix', Me50)
np.savetxt('Me100-matrix', Me100)

#EOF
