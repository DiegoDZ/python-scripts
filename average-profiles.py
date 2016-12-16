# -*- coding: utf-8 -*-
"""
Author: @DiegoDZ

Date: 24 may 2016
Modified: 13 june 2016

"""

"""
This script takes the mesoscopic files from CG-method and averages them.

Initial parameters: 	number_simulations

run: python average-profiles.py

"""

import os
import numpy as np


#Parameters
number_simulations = 100
number_snapshots = len(np.loadtxt('sim.MesoDensity_2.dat.1'))
number_nodes =len((np.loadtxt('sim.MesoDensity_2.dat.1')[0])) 

# Create arrays for outputs 
# Mesoscopic output
mesoDensity_fluid = np.zeros((number_snapshots, number_nodes))
mesoDensity_solid = np.zeros((number_snapshots, number_nodes))
mesoInternalEnergy = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_x = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_y = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_z = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_x = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_y = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_z = np.zeros((number_snapshots, number_nodes))

# Macroscopic output 
centerOfMass_UpperWall = np.zeros((number_snapshots, 3))
centerOfMass_LowerWall = np.zeros((number_snapshots, 3))
macroMomentum_UpperWall = np.zeros((number_snapshots, 3))
macroMomentum_LowerWall = np.zeros((number_snapshots, 3))
macroInternalEnergy_UpperWall = np.zeros(number_snapshots)
macroInternalEnergy_LowerWall = np.zeros(number_snapshots)

# Heat flux output
#mesoQ1_x = np.zeros((number_snapshots, number_nodes))
#mesoQ1_y = np.zeros((number_snapshots, number_nodes))
#mesoQ1_z = np.zeros((number_snapshots, number_nodes))
#mesoQ2_x = np.zeros((number_snapshots, number_nodes))
#mesoQ2_y = np.zeros((number_snapshots, number_nodes))
#mesoQ2_z = np.zeros((number_snapshots, number_nodes))
#mesoQ_x = np.zeros((number_snapshots, number_nodes))
#mesoQ_y = np.zeros((number_snapshots, number_nodes))
mesoQ_z = np.zeros((number_snapshots, number_nodes))
mesoPi = np.zeros((number_snapshots, number_nodes))


for file_name in os.listdir('./'):
# Mesoscopic output
    if file_name[0:21] == "sim.MesoDensity_2.dat":
        mesoDensity_fluid += np.loadtxt(file_name) / number_simulations
    if file_name[0:21] == "sim.MesoDensity_1.dat":
        mesoDensity_solid += np.loadtxt(file_name) / number_simulations
    elif file_name[0:26] == "sim.MesoInternalEnergy.dat":
        mesoInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoVelocity_0.dat":
        mesoVelocity_fluid_x += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoVelocity_1.dat":
        mesoVelocity_fluid_y += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoVelocity_2.dat":
        mesoVelocity_fluid_z += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoMomentum_0.dat":
        mesoMomentum_fluid_x += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoMomentum_1.dat":
        mesoMomentum_fluid_y += np.loadtxt(file_name) / number_simulations
    elif file_name[0:22] == "sim.MesoMomentum_2.dat":
        mesoMomentum_fluid_z += np.loadtxt(file_name) / number_simulations
# Macroscopic output 
    elif file_name[0:29] == "sim.CenterOfMassUpperWall.dat":
	centerOfMass_UpperWall += np.loadtxt(file_name, usecols = (1, 2, 3)) / number_simulations
    elif file_name[0:29] == "sim.CenterOfMassLowerWall.dat":
	centerOfMass_LowerWall += np.loadtxt(file_name, usecols = (1, 2, 3)) / number_simulations
    elif file_name[0:30] == "sim.MacroMomentumUpperWall.dat":
        macroMomentum_UpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:30] == "sim.MacroMomentumLowerWall.dat":
        macroMomentum_LowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36] == "sim.MacroInternalEnergyUpperWall.dat":
        macroInternalEnergy_UpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36] == "sim.MacroInternalEnergyLowerWall.dat":
        macroInternalEnergy_LowerWall += np.loadtxt(file_name) / number_simulations
# Heat flux output
    #elif file_name[0:16] == "sim.MesoQ1_x.dat": 
    #    mesoQ1_x += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:16] == "sim.MesoQ1_y.dat": 
    #    mesoQ1_y += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:16] == "sim.MesoQ1_z.dat": 
    #    mesoQ1_z += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:16] == "sim.MesoQ2_x.dat": 
    #    mesoQ2_x += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:16] == "sim.MesoQ2_y.dat": 
    #    mesoQ2_y += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:16] == "sim.MesoQ2_z.dat": 
    #    mesoQ2_z += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:15] == "sim.MesoQ_x.dat": 
    #    mesoQ_x += np.loadtxt(file_name) / number_simulations
    #elif file_name[0:15] == "sim.MesoQ_y.dat": 
    #    mesoQ_y += np.loadtxt(file_name) / number_simulations
    elif file_name[0:15] == "sim.MesoQ_z.dat": 
        mesoQ_z += np.loadtxt(file_name) / number_simulations
    elif file_name[0:14] == "sim.MesoPi.dat": 
        mesoPi += np.loadtxt(file_name) / number_simulations

# Save mesoscopic output
np.savetxt('mesoDensity_fluid', mesoDensity_fluid)
np.savetxt('mesoDensity_solid', mesoDensity_solid)
np.savetxt('mesoInternalEnergy', mesoInternalEnergy)
np.savetxt('mesoVelocity_fluid_x', mesoVelocity_fluid_x)
np.savetxt('mesoVelocity_fluid_y', mesoVelocity_fluid_y)
np.savetxt('mesoVelocity_fluid_z', mesoVelocity_fluid_z)
np.savetxt('mesoMomentum_fluid_x', mesoMomentum_fluid_x)
np.savetxt('mesoMomentum_fluid_y', mesoMomentum_fluid_y)
np.savetxt('mesoMomentum_fluid_z', mesoMomentum_fluid_z)

# Save macroscopic output
np.savetxt('centerOfMass_UpperWall', centerOfMass_UpperWall)
np.savetxt('centerOfMass_LowerWall', centerOfMass_LowerWall)
np.savetxt('macroMomentum_UpperWall', macroMomentum_UpperWall)
np.savetxt('macroMomentum_LowerWall', macroMomentum_LowerWall)
np.savetxt('macroInternalEnergy_UpperWall', macroInternalEnergy_UpperWall)
np.savetxt('macroInternalEnergy_LowerWall', macroInternalEnergy_LowerWall)

# Save heat flux 
#np.savetxt('mesoQ1_x', mesoQ1_x)
#np.savetxt('mesoQ1_y', mesoQ1_y)
#np.savetxt('mesoQ1_z', mesoQ1_z)
#np.savetxt('mesoQ2_x', mesoQ2_x)
#np.savetxt('mesoQ2_y', mesoQ2_y)
#np.savetxt('mesoQ2_z', mesoQ2_z)
#np.savetxt('mesoQ_x', mesoQ_x)
#np.savetxt('mesoQ_y', mesoQ_y)
np.savetxt('mesoQ_z', mesoQ_z)
np.savetxt('mesoPi', mesoPi)

#EOF
