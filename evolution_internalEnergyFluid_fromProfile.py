################
# This scripts computes the evolution of the internal energy of the fluid
################

import numpy as np

E = np.loadtxt('mesoInternal_energy_fluid')
number_snapshots = len(E)
number_nodes = len(E[0])
dt = 0.005

E_evolution = np.zeros((number_snapshots, number_nodes))
for t in range(1,number_snapshots):
    for i in range(0,number_nodes):
        E_evolution[t,i] = (E[t,i] - E[t-1,i]) / dt

np.savetxt('EVOLUTION_internalEnergyFluid', E_evolution)




