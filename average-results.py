# -*- coding: utf-8 -*-
"""
Author: @DiegoDZ

Date: 24 june 2016

"""

"""
This script takes the covariances files and averages them.

Initial parameters: 	number_simulations
                    	number_nodes

run: python stat.py

"""

import os
import numpy as np

number_simulations = 200
number_nodes = 100

# Create arrays for outputs. Matrices.
covariance_density = np.zeros((number_nodes, number_nodes))
covariance_densityInternalEnergy = np.zeros((number_nodes, number_nodes))
covariance_internalEnergy = np.zeros((number_nodes, number_nodes))
covariance_internalEnergyDensity = np.zeros((number_nodes, number_nodes))
# Create arrays for outputs. Vectors. 
covariance_densityCenterOfMassUpperWall = np.zeros(number_nodes)
covariance_internalEnergyCenterOfMassUpperWall = np.zeros(number_nodes)
covariance_densityCenterOfMassLowerWall = np.zeros(number_nodes)
covariance_internalEnergyCenterOfMassLowerWall = np.zeros(number_nodes)
covariance_densityMacroInternalEnergyUpperWall = np.zeros(number_nodes)
covariance_internalEnergyMacroInternalEnergyUpperWall = np.zeros(number_nodes)
covariance_densityMacroInternalEnergyLowerWall = np.zeros(number_nodes)
covariance_internalEnergyMacroInternalEnergyLowerWall = np.zeros(number_nodes)
covariance_centerOfMassLowerWallDensity = np.zeros(number_nodes) 
covariance_centerOfMassUpperWallDensity= np.zeros(number_nodes) 
covariance_macroInternalEnergyLowerWallDensity = np.zeros(number_nodes)
covariance_macroInternalEnergyUpperWallDensity = np.zeros(number_nodes)
covariance_centerOfMassLowerWallInternalEnergy = np.zeros(number_nodes)
covariance_centerOfMassUpperWallInternalEnergy = np.zeros(number_nodes)
covariance_macroInternalEnergyLowerWallInternalEnergy = np.zeros(number_nodes)
covariance_macroInternalEnergyUpperWallInternalEnergy = np.zeros(number_nodes)
# Create array one component. Numbers. 
covariance_centerOfMassLowerWall = np.zeros(1)
covariance_macroInternalEnergyLowerWall = np.zeros(1)
covariance_macroInternalEnergyCenterOfMassLowerWall = np.zeros(1)
covariance_centerOfMassUpperWall = np.zeros(1)
covariance_macroInternalEnergyUpperWall = np.zeros(1)
covariance_macroInternalEnergyCenterOfMassUpperWall = np.zeros(1)
covariance_centerOfMassUpperLowerWall = np.zeros(1)
covariance_macroInternalEnergyLowerWallCenterOfMassUpperWall = np.zeros(1)
covariance_macroInternalEnergyUpperWallCenterOfMassLowerWall = np.zeros(1)
covariance_macroInternalEnergyUpperLowerWall = np.zeros(1)

for file_name in os.listdir('./'):
    if file_name[0:1] == "A":
        covariance_density += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "B":
        covariance_densityInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "C":
        covariance_internalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "D":
        covariance_densityCenterOfMassUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "E":
        covariance_internalEnergyCenterOfMassUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "F":
        covariance_densityCenterOfMassLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "G":
        covariance_internalEnergyCenterOfMassLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "H":
	    covariance_densityMacroInternalEnergyUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "I":
	    covariance_internalEnergyMacroInternalEnergyUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "J":
        covariance_densityMacroInternalEnergyLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "K":
        covariance_internalEnergyMacroInternalEnergyLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "L":
        covariance_centerOfMassLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "M":
        covariance_macroInternalEnergyLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "N":
        covariance_macroInternalEnergyCenterOfMassLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "O":
        covariance_centerOfMassUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "P":
        covariance_macroInternalEnergyUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "Q":
        covariance_macroInternalEnergyCenterOfMassUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "R":
	    covariance_internalEnergyDensity += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "S":
        covariance_centerOfMassLowerWallDensity += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "T":
        covariance_centerOfMassUpperWallDensity += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "U":
        covariance_macroInternalEnergyLowerWallDensity += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "V":
        covariance_macroInternalEnergyUpperWallDensity += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "W":
        covariance_centerOfMassLowerWallInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "X":
        covariance_centerOfMassUpperWallInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "Y":
        covariance_macroInternalEnergyLowerWallInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "Z":
        covariance_macroInternalEnergyUpperWallInternalEnergy += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "f":
        covariance_centerOfMassUpperLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "g":
        covariance_macroInternalEnergyLowerWallCenterOfMassUpperWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "e":
        covariance_macroInternalEnergyUpperWallCenterOfMassLowerWall += np.loadtxt(file_name) / number_simulations
    elif file_name[0:1] == "d":
        covariance_macroInternalEnergyUpperLowerWall += np.loadtxt(file_name) / number_simulations
   
# Save output

np.savetxt('covariance_density', covariance_density)
np.savetxt('covariance_densityInternalEnergy', covariance_densityInternalEnergy)
np.savetxt('covariance_internalEnergy', covariance_internalEnergy)
np.savetxt('covariance_densityCenterOfMassUpperWall', covariance_densityCenterOfMassUpperWall)
np.savetxt('covariance_internalEnergyCenterOfMassUpperWall', covariance_internalEnergyCenterOfMassUpperWall)
np.savetxt('covariance_densityCenterOfMassLowerWall', covariance_densityCenterOfMassLowerWall)
np.savetxt('covariance_internalEnergyCenterOfMassLowerWall', covariance_internalEnergyCenterOfMassLowerWall)
np.savetxt('covariance_densityMacroInternalEnergyUpperWall', covariance_densityMacroInternalEnergyUpperWall)
np.savetxt('covariance_internalEnergyMacroInternalEnergyUpperWall', covariance_internalEnergyMacroInternalEnergyUpperWall)
np.savetxt('covariance_densityMacroInternalEnergyLowerWall', covariance_densityMacroInternalEnergyLowerWall)
np.savetxt('covariance_internalEnergyMacroInternalEnergyLowerWall', covariance_internalEnergyMacroInternalEnergyLowerWall)
np.savetxt('covariance_centerOfMassLowerWall', covariance_centerOfMassLowerWall)
np.savetxt('covariance_macroInternalEnergyLowerWall', covariance_macroInternalEnergyLowerWall)
np.savetxt('covariance_macroInternalEnergyCenterOfMassLowerWall', covariance_macroInternalEnergyCenterOfMassLowerWall)
np.savetxt('covariance_centerOfMassUpperWall', covariance_centerOfMassUpperWall)
np.savetxt('covariance_macroInternalEnergyUpperWall', covariance_macroInternalEnergyUpperWall)
np.savetxt('covariance_macroInternalEnergyCenterOfMassUpperWall', covariance_macroInternalEnergyCenterOfMassUpperWall)
np.savetxt('covariance_internalEnergyDensity', covariance_internalEnergyDensity)
np.savetxt('covariance_centerOfMassLowerWallDensity', covariance_centerOfMassLowerWallDensity)
np.savetxt('covariance_centerOfMassUpperWallDensity', covariance_centerOfMassUpperWallDensity)
np.savetxt('covariance_macroInternalEnergyLowerWallDensity', covariance_macroInternalEnergyLowerWallDensity)
np.savetxt('covariance_macroInternalEnergyUpperWallDensity', covariance_macroInternalEnergyUpperWallDensity)
np.savetxt('covariance_centerOfMassLowerWallInternalEnergy', covariance_centerOfMassLowerWallInternalEnergy)
np.savetxt('covariance_centerOfMassUpperWallInternalEnergy', covariance_centerOfMassUpperWallInternalEnergy)
np.savetxt('covariance_macroInternalEnergyLowerWallInternalEnergy', covariance_macroInternalEnergyLowerWallInternalEnergy)
np.savetxt('covariance_macroInternalEnergyUpperWallInternalEnergy', covariance_macroInternalEnergyUpperWallInternalEnergy)
np.savetxt('covariance_centerOfMassUpperLowerWall', covariance_centerOfMassUpperLowerWall)
np.savetxt('covariance_macroInternalEnergyLowerWallCenterOfMassUpperWall', covariance_macroInternalEnergyLowerWallCenterOfMassUpperWall)
np.savetxt('covariance_macroInternalEnergyUpperWallCenterOfMassLowerWall', covariance_macroInternalEnergyUpperWallCenterOfMassLowerWall)
np.savetxt('covariance_macroInternalEnergyUpperLowerWall', covariance_macroInternalEnergyUpperLowerWall)


#EOF
