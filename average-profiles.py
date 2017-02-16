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
mesoDensity_wall = np.zeros((number_snapshots, number_nodes))
mesoInternalEnergy_fluid = np.zeros((number_snapshots, number_nodes))
mesoDerivativeInternalEnergy_fluid = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_x = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_y = np.zeros((number_snapshots, number_nodes))
mesoVelocity_fluid_z = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_x = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_y = np.zeros((number_snapshots, number_nodes))
mesoMomentum_fluid_z = np.zeros((number_snapshots, number_nodes))

# Macroscopic output
centerOfMass_upperWall = np.zeros((number_snapshots, 3))
centerOfMass_lowerWall = np.zeros((number_snapshots, 3))
macroMomentum_upperWall = np.zeros((number_snapshots, 3))
macroMomentum_lowerWall = np.zeros((number_snapshots, 3))
macroInternalEnergy_upperWall = np.zeros(number_snapshots)
macroInternalEnergy_lowerWall = np.zeros(number_snapshots)

#macroInternalEnergy_wall = np.zeros(number_snapshots)

# Heat flux output
mesoQ_z = np.zeros((number_snapshots, number_nodes))
mesoPi = np.zeros((number_snapshots, number_nodes))

# Outputs for averages over time
mesoDensity_fluid_avg = np.zeros((number_snapshots, number_nodes))
mesoDensity_wall_avg = np.zeros((number_snapshots, number_nodes))
mesoInternalEnergy_fluid_avg = np.zeros((number_snapshots, number_nodes))
mesoDerivativeInternalEnergy_fluid_avg = np.zeros((number_snapshots, number_nodes))
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
        mesoDensity_wall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:26] == "sim.MesoInternalEnergy.dat":
        mesoInternalEnergy_fluid += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36]  == "sim.MesoDerivativeInternalEnergy.dat":
        mesoDerivativeInternalEnergy_fluid += np.loadtxt(file_name) / number_simulations
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
	centerOfMass_upperWall += np.loadtxt(file_name, usecols = (1, 2, 3)) / number_simulations
    elif file_name[0:29] == "sim.CenterOfMassLowerWall.dat":
	centerOfMass_lowerWall += np.loadtxt(file_name, usecols = (1, 2, 3)) / number_simulations
    elif file_name[0:30] == "sim.MacroMomentumUpperWall.dat":
        macroMomentum_upperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:30] == "sim.MacroMomentumLowerWall.dat":
        macroMomentum_lowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36] == "sim.MacroInternalEnergyUpperWall.dat":
        macroInternalEnergy_upperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:36] == "sim.MacroInternalEnergyLowerWall.dat":
        macroInternalEnergy_lowerWall += np.loadtxt(file_name) / number_simulations
# Heat flux output
    elif file_name[0:15] == "sim.MesoQ_z.dat":
        mesoQ_z += np.loadtxt(file_name) / number_simulations
    elif file_name[0:14] == "sim.MesoPi.dat":
        mesoPi += np.loadtxt(file_name) / number_simulations

#####################
# Sum macroInternalEnergy_lowerWall and macroInternalEnergy_upperWall
#####################
macroInternalEnergy_wall = macroInternalEnergy_lowerWall + macroInternalEnergy_upperWall

#####################
# Average over time
#####################
mesoDensity_fluid_avg = np.sum(mesoDensity_fluid, axis = 0) / number_snapshots
mesoDensity_wall_avg = np.sum(mesoDensity_wall, axis = 0) /number_snapshots
mesoInternalEnergy_fluid_avg = np.sum(mesoInternalEnergy_fluid, axis = 0) / number_snapshots
mesoDerivativeInternalEnergy_fluid_avg = np.sum(mesoDerivativeInternalEnergy_fluid, axis = 0) /number_snapshots
mesoQ_z_avg = np.sum(mesoQ_z, axis = 0) /number_snapshots
mesoPi_avg = np.sum(mesoPi, axis = 0) /number_snapshots

####################
# Save output files
####################
# Save mesoscopic output
np.savetxt('mesoDensity_fluid', mesoDensity_fluid)
np.savetxt('mesoDensity_wall', mesoDensity_wall)
np.savetxt('mesoInternalEnergy_fluid', mesoInternalEnergy_fluid)
np.savetxt('mesoDerivativeInternalEnergy_fluid', mesoDerivativeInternalEnergy_fluid)
np.savetxt('mesoVelocity_fluid_x', mesoVelocity_fluid_x)
np.savetxt('mesoVelocity_fluid_y', mesoVelocity_fluid_y)
np.savetxt('mesoVelocity_fluid_z', mesoVelocity_fluid_z)
np.savetxt('mesoMomentum_fluid_x', mesoMomentum_fluid_x)
np.savetxt('mesoMomentum_fluid_y', mesoMomentum_fluid_y)
np.savetxt('mesoMomentum_fluid_z', mesoMomentum_fluid_z)

# Save macroscopic output
np.savetxt('centerOfMass_upperWall', centerOfMass_upperWall)
np.savetxt('centerOfMass_lowerWall', centerOfMass_lowerWall)
np.savetxt('macroMomentum_upperWall', macroMomentum_upperWall)
np.savetxt('macroMomentum_lowerWall', macroMomentum_lowerWall)
np.savetxt('macroInternalEnergy_upperWall', macroInternalEnergy_upperWall)
np.savetxt('macroInternalEnergy_lowerWall', macroInternalEnergy_lowerWall)
np.savetxt('macroInternalEnergy_wall', macroInternalEnergy_wall)

# Save heat flux
np.savetxt('mesoQ_z', mesoQ_z)
np.savetxt('mesoPi', mesoPi)

# Save averages over time
np.savetxt('mesoDensity_fluid-avg', mesoDensity_fluid_avg)
np.savetxt('mesoDensity_wall-avg', mesoDensity_wall_avg)
np.savetxt('mesoInternalEnergy_fluid-avg', mesoInternalEnergy_fluid_avg)
np.savetxt('mesoDerivativeInternalEnergy_fluid-avg', mesoDerivativeInternalEnergy_fluid_avg)
np.savetxt('mesoQ_z-avg', mesoQ_z_avg)
np.savetxt('mesoPi-avg', mesoPi_avg)

#EOF
