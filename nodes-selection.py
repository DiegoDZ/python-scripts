# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:22:00 2016



@author: DiegoDZ

run: >> python nodes-selection.py

"""

import numpy as np

#Load files
mesoInternalEnergy = np.loadtxt('sim.MesoInternalEnergy.dat')
Q = np.loadtxt('sim.MesoQ_z.dat')
Pi = np.loadtxt('sim.MesoPi.dat')
iLe = np.loadtxt('sim.MesoDerivativeInternalEnergy.dat')

#Delete useless nodes (vacuum and solid)
number_nodes = len(Q[0])
number_zeros_upper = 34
number_zeros_lower = 33
mesoInternalEnergy = mesoInternalEnergy[:, number_zeros_lower: number_nodes - number_zeros_upper]
Q = Q[:, number_zeros_lower: number_nodes - number_zeros_upper]
Pi = Pi[:, number_zeros_lower: number_nodes - number_zeros_upper]
iLe = iLe[:, number_zeros_lower: number_nodes - number_zeros_upper]

#Save output
np.savetxt('InternalEnergy', mesoInternalEnergy)
np.savetxt('Q', Q)
np.savetxt('Pi', Pi)
np.savetxt('iLe', iLe)
