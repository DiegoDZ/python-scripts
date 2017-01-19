#Computes the evolution of the internal energy of the fluid from the dissipative matrix, D, and the inverse of the temperature, beta.

import numpy as np

#Load files
D = np.loadtxt('Me7-matrix')
beta = np.loadtxt('betaFinal')

#Parameters
number_nodes = len(beta[0])
number_snapshots = len(beta)

#Arrays
internalEnergy_evolution = np.zeros([number_snapshots, number_nodes])

#Calculate the product D * beta
for i in range(number_snapshots):
    internalEnergy_evolution[i,:] = np.dot(-D,beta[i,:])

#Save the output
np.savetxt('internalEnergy_evolution', internalEnergy_evolution)
