# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:36:25 2016

Author: DiegoDZ

Compute matrix D

"""

import numpy as np

#Load time correlation files
C_iLrhoiLrho = np.loadtxt('corr_iLrhoiLrho')
C_iLrhoiLe = np.loadtxt('corr_iLrhoiLe')
C_iLeiLrho = np.loadtxt('corr_iLeiLrho')
C_iLeiLe = np.loadtxt('corr_iLeiLe')

#Select the range of snapshots will be taken into account.
C_iLrhoiLrho7 = C_iLrhoiLrho[0:7, :]
C_iLrhoiLrho20 = C_iLrhoiLrho[0:20, :]
C_iLrhoiLrho50 = C_iLrhoiLrho[0:50, :]
C_iLrhoiLe7 = C_iLrhoiLe[0:7, :]
C_iLrhoiLe20 = C_iLrhoiLe[0:20, :]
C_iLrhoiLe50 = C_iLrhoiLe[0:50, :]
C_iLeiLrho7 = C_iLeiLrho[0:7, :]
C_iLeiLrho20 = C_iLeiLrho[0:20, :]
C_iLeiLrho50 = C_iLeiLrho[0:50, :]
C_iLeiLe7 = C_iLeiLe[0:7, :]
C_iLeiLe20 = C_iLeiLe[0:20, :]
C_iLeiLe50 = C_iLeiLe[0:50, :]

#Define variables
#Box length
Lx = 17.3162
Ly = 17.3162
Lz = 34.6325
#Number nodes
number_nodes = 100
#Number nodes fluid
number_nodes_fluid = np.sqrt(len(C_iLrhoiLrho[0]))
#binSize = Lz / number_nodes
dz = Lz / number_nodes
#Bin volume
V = dz * Lx * Ly
#Time step
dt = 0.005

# Compute component of matrix D from iLerho and iLepsilon
rhorho7 = V * (np.sum(C_iLrhoiLrho7, axis = 0))
Drhorho7  = rhorho7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
rhorho20 =  V * (np.sum(C_iLrhoiLrho20, axis = 0))
Drhorho20  = rhorho20.reshape(number_nodes_fluid, number_nodes_fluid) * dt
rhorho50 =  V * (np.sum(C_iLrhoiLrho50, axis = 0))
Drhorho50  = rhorho50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
rhoe7 = V * (np.sum(C_iLrhoiLe7, axis = 0))
Drhoe7  = rhoe7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
rhoe20 =  V * (np.sum(C_iLrhoiLe20, axis = 0))
Drhoe20  = rhoe20.reshape(number_nodes_fluid, number_nodes_fluid) * dt
rhoe50 =  V * (np.sum(C_iLrhoiLe50, axis = 0))
Drhoe50  = rhoe50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
erho7 = V * (np.sum(C_iLeiLrho7, axis = 0))
Derho7  = erho7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
erho20 =  V * (np.sum(C_iLeiLrho20, axis = 0))
Derho20  = erho20.reshape(number_nodes_fluid, number_nodes_fluid) * dt
erho50 =  V * (np.sum(C_iLeiLrho50, axis = 0))
Derho50  = erho50.reshape(number_nodes_fluid, number_nodes_fluid) * dt
ee7 = V * (np.sum(C_iLeiLe7, axis = 0))
Dee7  = ee7.reshape(number_nodes_fluid, number_nodes_fluid) * dt
ee20 =  V * (np.sum(C_iLeiLe20, axis = 0))
Dee20  = ee20.reshape(number_nodes_fluid, number_nodes_fluid) * dt
ee50 =  V * (np.sum(C_iLeiLe50, axis = 0))
Dee50  = ee50.reshape(number_nodes_fluid, number_nodes_fluid) * dt

# Transform the arrays into matrix

Drhorho7 = np.asmatrix(Drhorho7)  
Drhorho20 = np.asmatrix(Drhorho20)
Drhorho50 = np.asmatrix(Drhorho50)
Drhoe7   = np.asmatrix(Drhoe7)
Drhoe20  = np.asmatrix(Drhoe20)
Drhoe50  = np.asmatrix(Drhoe50)
Derho7   = np.asmatrix(Derho7)
Derho20  = np.asmatrix(Derho20)
Derho50  = np.asmatrix(Derho50)
Dee7   = np.asmatrix(Dee7)
Dee20  = np.asmatrix(Dee20)
Dee50  = np.asmatrix(Dee50)

# Create the matrix D
D7 = np.bmat([[Drhorho7, Drhoe7],[Derho7, Dee7]])
D20 = np.bmat([[Drhorho20, Drhoe20],[Derho20, Dee20]])
D50 = np.bmat([[Drhorho50, Drhoe50],[Derho50, Dee50]])

#Save D-matrix
np.savetxt('D7-matrix', D7)
np.savetxt('D20-matrix', D20)
np.savetxt('D50-matrix', D50)
#Save components D-matrix
np.savetxt('Drhorho7', Drhorho7)
np.savetxt('Drhorho20', Drhorho20)
np.savetxt('Drhorho50', Drhorho50)
np.savetxt('Drhoe7', Drhoe7)
np.savetxt('Drhoe20', Drhoe20)
np.savetxt('Drhoe50', Drhoe50)
np.savetxt('Derho7', Derho7)
np.savetxt('Derho20', Derho20)
np.savetxt('Derho50', Derho50)
np.savetxt('Dee7', Dee7)
np.savetxt('Dee20', Dee20)
np.savetxt('Dee50', Dee50)

#EOF
