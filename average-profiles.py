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
number_simulations = 200
number_snapshots = len(np.loadtxt('sim.MesoDensity_2.dat.1'))
number_nodes =len((np.loadtxt('sim.MesoDensity_2.dat.1')[0])) 

############################
# Create arrays for outputs 
############################
# Mesoscopic output
mesoDensity_fluid = np.zeros((number_snapshots, number_nodes))
mesoDensity_solid = np.zeros((number_snapshots, number_nodes))
mesoInternalEnergy = np.zeros((number_snapshots, number_nodes))
mesoDerivativeInternalEnergy = np.zeros((number_snapshots, number_nodes))
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
mesoQ_z = np.zeros((number_snapshots, number_nodes))
mesoPi = np.zeros((number_snapshots, number_nodes))

# Outputs for averages over time
mesoDensity_fluid_avg = np.zeros((number_snapshots, number_nodes))
mesoDensity_solid_avg = np.zeros((number_snapshots, number_nodes))
mesoInternalEnergy_avg = np.zeros((number_snapshots, number_nodes))
mesoDerivativeInternalEnergy_avg = np.zeros((number_snapshots, number_nodes))
mesoQ_z_avg = np.zeros((number_snapshots, number_nodes))
mesoPi_avg = np.zeros((number_snapshots, number_nodes))

###########################
# Average over simulations
###########################
for file_name in os.listdir('./'):
# Mesoscopic output
    if file_name[0:21] == "sim.MesoDensity_2.dat":
        mesoDensity_fluid += np.loadtxt(file_name) / number_simulations
    if file_name[0:21] == "sim.MesoDensity_1.dat":
        mesoDensity_solid += np.loadtxt(file_name) / number_simulations
    elif file_name[0:26] == "sim.MesoInternalEnergy.dat":
        mesoInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36]  == "sim.MesoDerivativeInternalEnergy.dat":
        mesoDerivativeInternalEnergy += np.loadtxt(file_name) / number_simulations
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
    elif file_name[0:15] == "sim.MesoQ_z.dat": 
        mesoQ_z += np.loadtxt(file_name) / number_simulations
    elif file_name[0:14] == "sim.MesoPi.dat": 
        mesoPi += np.loadtxt(file_name) / number_simulations

#####################
# Average over time
#####################
mesoDensity_fluid_avg = np.sum(mesoDensity_fluid, axis = 0) / number_snapshots
mesoDensity_solid_avg = np.sum(mesoDensity_solid, axis = 0) /number_snapshots
mesoInternalEnergy_avg = np.sum(mesoInternalEnergy, axis = 0) / number_snapshots
mesoDerivativeInternalEnergy_avg = np.sum(mesoDerivativeInternalEnergy, axis = 0) /number_snapshots
mesoQ_z_avg = np.sum(mesoQ_z, axis = 0) /number_snapshots
mesoPi_avg = np.sum(mesoPi, axis = 0) /number_snapshots

####################
# Save output files
####################
# Save mesoscopic output
np.savetxt('mesoDensity_fluid', mesoDensity_fluid)
np.savetxt('mesoDensity_solid', mesoDensity_solid)
np.savetxt('mesoInternalEnergy', mesoInternalEnergy)
np.savetxt('mesoDerivativeInternalEnergy', mesoDerivativeInternalEnergy)
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
np.savetxt('mesoQ_z', mesoQ_z)
np.savetxt('mesoPi', mesoPi)

# Save averages over time
np.savetxt('mesoDensity_fluid-avg', mesoDensity_fluid_avg)
np.savetxt('mesoDensity_solid-avg', mesoDensity_solid_avg)
np.savetxt('mesoInternalEnergy-avg', mesoInternalEnergy_avg)
np.savetxt('mesoDerivativeInternalEnergy-avg', mesoDerivativeInternalEnergy_avg)
np.savetxt('mesoQ_z-avg', mesoQ_z_avg)
np.savetxt('mesoPi-avg', mesoPi_avg)

#EOF
