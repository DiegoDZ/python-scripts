# -*- coding: utf-8 -*-
"""

This script creates the matrix of covariances, C, and its inverse, H. With H and the mesoscopic and macroscopic profiles, it computes the conjugates variables.

Created on Thu Jul 14 10:32:27 2016

@author: DiegoDZ
"""
import numpy as np
import os
import shutil

##########################################################
# Compute the matrix of covariances, C, and its inverse, H
##########################################################

# Looking one profile output file: Number of zero rows until the first row with at least one element != 0
number_zeros_lower = 16
# Looking one profile output file: Number of zero rows after the last row with at least one element != 0
number_zeros_upper = 17

# Load covariances files
c00 = np.loadtxt('covariance_density')
c01 = np.loadtxt('covariance_densityMesoEnergy')
c02=  np.loadtxt('covariance_densityMacroEnergy')
c10 = np.loadtxt('covariance_mesoEnergyDensity')
c11 = np.loadtxt('covariance_mesoEnergy')
c12 = np.loadtxt('covariance_mesoEnergyMacroEnergy')
c20 = np.loadtxt('covariance_macroEnergyDensity')
c21 = np.loadtxt('covariance_macroEnergyMesoEnergy')
c22 = np.loadtxt('covariance_macroEnergy')

# Delete zero elements. In this stage we use number_zeros_lower,number_zeros_upper and number_nodes.
number_nodes = len(c00) # All files have the same number of rows, so it is not necessary to compute the lenght of all files
c00 = c00[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
c01 = c01[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
c02 = c02[number_zeros_lower:number_nodes-number_zeros_upper]
c10 = c10[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
c11 = c11[number_zeros_lower:number_nodes-number_zeros_upper, number_zeros_lower:number_nodes-number_zeros_upper]
c12 = c12[number_zeros_lower:number_nodes-number_zeros_upper]
c20 = c20[number_zeros_lower:number_nodes-number_zeros_upper]
c21 = c21[number_zeros_lower:number_nodes-number_zeros_upper]

# Transform arrays into matrix
C00 = np.asmatrix(c00)
C01 = np.asmatrix(c01)
C02 = np.asmatrix(c02)
C10 = np.asmatrix(c10)
C11 = np.asmatrix(c11)
C12 = np.asmatrix(c12)
C20 = np.asmatrix(c20)
C21 = np.asmatrix(c21)
C22 = np.asmatrix(c22)

# Create the matrix of covariances, C
C = np.bmat([[C00, C01, C02.T],[C10, C11, C12.T],[C20, C21, C22]])

# Compute the inverse of C, H
H = C.I

# Save the matrix H
np.savetxt('H', H)

########################################################################################
# Compute conjugate variables (mu, beta, lambdaUpper, lambdaLower, betaUpper, betaLower)
########################################################################################

# Load profile files.
densityFluid = np.asmatrix(np.loadtxt('mesoDensity_fluid'))
energyFluid = np.asmatrix(np.loadtxt('mesoEnergy_fluid'))
energyWall = np.asmatrix(np.loadtxt('macroEnergy_wall'))

# Delete zero elements which corresponds to walls nodes.
densityFluid = densityFluid[:,number_zeros_lower:number_nodes-number_zeros_upper]
energyFluid = energyFluid[:, number_zeros_lower:number_nodes-number_zeros_upper]

# Introduce new variable called nodes: number of nodes in which the elements are !=0.
nodes = number_nodes - (number_zeros_lower + number_zeros_upper)
# Number snapshots
number_snapshots = len(densityFluid)

# Subtract the last row (system in equilibrium) to all rows: fluctuation of CG variables.
densityFluid = densityFluid - densityFluid[number_snapshots-1,:]
energyFluid = energyFluid - energyFluid[number_snapshots-1,:]
energyWall = energyWall - energyWall[:,number_snapshots-1]

"""
####
# Compute terms of mu (chemical potential)
####

mu_term_density = np.zeros((number_snapshots,nodes))
mu_term_energy = np.zeros((number_snapshots,nodes))
for t in range(0,number_snapshots):
    for i in range(0,nodes):
        mu_term_density[t,i] = np.sum(H[i,0:nodes] * densityFluid[t,:].T)
        mu_term_energy[t,i] = np.sum(H[i,nodes:2*nodes] * internalEnergyFluid[t,:].T)
mu_term_cmUpper = np.zeros((number_snapshots,nodes))
mu_term_cmLower = np.zeros((number_snapshots,nodes))
mu_term_macroEnergyUpper = np.zeros((number_snapshots,nodes))
mu_term_macroEnergyLower = np.zeros((number_snapshots,nodes))
for i in range(0, nodes):
    mu_term_cmUpper[:,i] = H[i,2*nodes] * centerOfMassUpperWall[:,2].T
    mu_term_cmLower[:,i] = H[i,2*nodes+1] * centerOfMassLowerWall[:,2].T
    mu_term_macroEnergyUpper[:,i] = H[i,2*nodes+2] * macroInternalEnergyUpperWall
    mu_term_macroEnergyLower[:,i] = H[i,2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
mu = mu_term_density + mu_term_energy + mu_term_cmUpper + mu_term_cmLower + mu_term_macroEnergyUpper + mu_term_macroEnergyLower

####
# Compute terms of beta (temperature)
####
beta_term_density = np.zeros((number_snapshots, nodes))
beta_term_energy = np.zeros((number_snapshots, nodes))
for t in range(0, number_snapshots):
    for i in range(nodes, 2*nodes):
        beta_term_density[t,i-nodes] = np.sum(H[i,0:nodes] * densityFluid[t,:].T)
        beta_term_energy[t,i-nodes] = np.sum(H[i, nodes:2*nodes] * internalEnergyFluid[t,:].T)
beta_term_cmUpper = np.zeros((number_snapshots, nodes))
beta_term_cmLower = np.zeros((number_snapshots, nodes))
beta_term_macroEnergyUpper = np.zeros((number_snapshots, nodes))
beta_term_macroEnergyLower = np.zeros((number_snapshots, nodes))
for i in range(nodes, 2*nodes):
    beta_term_cmUpper[:,i-nodes] = H[i, 2*nodes] * centerOfMassUpperWall[:,2].T
    beta_term_cmLower[:,i-nodes] = H[i, 2*nodes+1] * centerOfMassLowerWall[:,2].T
    beta_term_macroEnergyUpper[:,i-nodes] = H[i, 2*nodes+2] * macroInternalEnergyUpperWall
    beta_term_macroEnergyLower[:,i-nodes] = H[i, 2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
beta = beta_term_density + beta_term_energy + beta_term_cmUpper + beta_term_cmLower + beta_term_macroEnergyUpper + beta_term_macroEnergyLower
"""

##############################
# Computes the terms of lambda
##############################
lambda_term_densityFluid = np.zeros((number_snapshots, nodes))
lambda_term_energyFluid = np.zeros((number_snapshots, nodes))
for t in range(0, number_snapshots):
    for i in range(0, nodes):
        lambda_term_densityFluid[t,i] = np.sum(H[i,0:nodes] * densityFluid[t,:].T)
        lambda_term_energyFluid[t,i] = np.sum(H[i, nodes:2*nodes] * energyFluid[t,:].T)

lambda_term_energyWall = np.zeros((number_snapshots, nodes))
for i in range(0, nodes):
    lambda_term_energyWall[:,i] = H[i, 2*nodes] * energyWall
# Sum terms
Lambda = lambda_term_energyFluid + lambda_term_energyWall
Lambda = np.asmatrix(Lambda)

###########################
#Compute the terms of beta
###########################
beta_term_densityFluid = np.zeros((number_snapshots, nodes))
beta_term_energyFluid = np.zeros((number_snapshots, nodes))
for t in range(0, number_snapshots):
    for i in range(nodes, 2*nodes):
        beta_term_densityFluid[t,i-nodes] = np.sum(H[i,0:nodes] * densityFluid[t,:].T)
        beta_term_energyFluid[t,i-nodes] = np.sum(H[i, nodes:2*nodes] * energyFluid[t,:].T)

beta_term_energyWall = np.zeros((number_snapshots, nodes))
for i in range(nodes, 2*nodes):
    beta_term_energyWall[:,i-nodes] = H[i, 2*nodes] * energyWall
# Sum terms
beta = beta_term_energyFluid + beta_term_energyWall
beta = np.asmatrix(beta)

############################
#Compute terms of betaWall
############################
betaWall_term_densityFluid = np.zeros(number_snapshots)
betaWall_term_energyFluid = np.zeros(number_snapshots)
for t in range(0, number_snapshots):
     betaWall_term_densityFluid[t] = np.sum(H[2*nodes, 0:nodes] * densityFluid[t,:].T)
     betaWall_term_energyFluid[t] = np.sum(H[2*nodes, nodes:2*nodes] * energyFluid[t,:].T)
betaWall_term_energyWall = H[2*nodes, 2*nodes] * energyWall
#Sum terms
betaWall = betaWall_term_densityFluid + betaWall_term_energyFluid + betaWall_term_energyWall
betaWall = np.asmatrix(betaWall)

"""
####
# Compute terms of lambdaUpper
####

lambdaUpper_term_density = np.zeros(number_snapshots)
lambdaUpper_term_energy = np.zeros(number_snapshots)
for t in range(0, number_snapshots):
     lambdaUpper_term_density[t] = np.sum(H[2*nodes, 0:nodes] * densityFluid[t,:].T)
     lambdaUpper_term_energy[t] = np.sum(H[2*nodes, nodes:2*nodes] * internalEnergyFluid[t,:].T)
lambdaUpper_term_cmUpper = H[2*nodes, 2*nodes] * centerOfMassUpperWall[:,2].T
lambdaUpper_term_cmLower = H[2*nodes, 2*nodes+1] * centerOfMassLowerWall[:,2].T
lambdaUpper_term_macroEnergyUpper = H[2*nodes, 2*nodes+2] * macroInternalEnergyUpperWall
lambdaUpper_term_macroEnergyLower = H[2*nodes, 2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
lambdaUpper = lambdaUpper_term_density + lambdaUpper_term_energy + lambdaUpper_term_cmUpper + lambdaUpper_term_cmLower + lambdaUpper_term_macroEnergyUpper + lambdaUpper_term_macroEnergyLower

####
# Compute terms of lambdaLower
####

lambdaLower_term_density = np.zeros(number_snapshots)
lambdaLower_term_energy = np.zeros(number_snapshots)
for t in range(0, number_snapshots):
     lambdaLower_term_density[t] = np.sum(H[2*nodes+1, 0:nodes] * densityFluid[t,:].T)
     lambdaLower_term_energy[t] = np.sum(H[2*nodes+1, nodes:2*nodes] * internalEnergyFluid[t,:].T)
lambdaLower_term_cmUpper = H[2*nodes+1, 2*nodes] * centerOfMassUpperWall[:,2].T
lambdaLower_term_cmLower = H[2*nodes+1, 2*nodes+1] * centerOfMassLowerWall[:,2].T
lambdaLower_term_macroEnergyUpper = H[2*nodes+1, 2*nodes+2] * macroInternalEnergyUpperWall
lambdaLower_term_macroEnergyLower = H[2*nodes+1, 2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
lambdaLower = lambdaLower_term_density + lambdaLower_term_energy + lambdaLower_term_cmUpper + lambdaLower_term_cmLower + lambdaLower_term_macroEnergyUpper + lambdaLower_term_macroEnergyLower

####
# Compute terms of betaUpper
####
#betaUpper_term_density = np.zeros(number_snapshots)
betaUpper_term_energy = np.zeros(number_snapshots)
for t in range(0, number_snapshots):
     #betaUpper_term_density[t] = np.sum(H[2*nodes+2, 0:nodes] * densityFluid[t,:].T)
     betaUpper_term_energy[t] = np.sum(H[2*nodes+2, nodes:2*nodes] * internalEnergyFluid[t,:].T)
#betaUpper_term_cmUpper = H[2*nodes+2, 2*nodes] * centerOfMassUpperWall[:,2].T
#betaUpper_term_cmLower = H[2*nodes+2, 2*nodes+1] * centerOfMassLowerWall[:,2].T
betaUpper_term_macroEnergyUpper = H[2*nodes+2, 2*nodes+2] * macroInternalEnergyUpperWall
betaUpper_term_macroEnergyLower = H[2*nodes+2, 2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
#betaUpper = betaUpper_term_density + betaUpper_term_energy + betaUpper_term_cmUpper + betaUpper_term_cmLower + betaUpper_term_macroEnergyUpper + betaUpper_term_macroEnergyLower
betaUpper = betaUpper_term_energy + betaUpper_term_macroEnergyUpper + betaUpper_term_macroEnergyLower

####
# Compute terms of betaLower
####
#betaLower_term_density = np.zeros(number_snapshots)
betaLower_term_energy = np.zeros(number_snapshots)
for t in range(0, number_snapshots):
     #betaLower_term_density[t] = np.sum(H[2*nodes+3, 0:nodes] * densityFluid[t,:].T)
     betaLower_term_energy[t] = np.sum(H[2*nodes+3, nodes:2*nodes] * internalEnergyFluid[t,:].T)
#betaLower_term_cmUpper = H[2*nodes+3, 2*nodes] * centerOfMassUpperWall[:,2].T
#betaLower_term_cmLower = H[2*nodes+3, 2*nodes+1] * centerOfMassLowerWall[:,2].T
betaLower_term_macroEnergyUpper = H[2*nodes+3, 2*nodes+2] * macroInternalEnergyUpperWall
betaLower_term_macroEnergyLower = H[2*nodes+3, 2*nodes+3] * macroInternalEnergyLowerWall
# Sum terms
#betaLower = betaLower_term_density + betaLower_term_energy + betaLower_term_cmUpper + betaLower_term_cmLower + betaLower_term_macroEnergyUpper + betaLower_term_macroEnergyLower
betaLower =  betaLower_term_energy + betaLower_term_macroEnergyUpper + betaLower_term_macroEnergyLower
"""

# Save terms
np.savetxt('lambda_term_densityFluid', lambda_term_densityFluid)
np.savetxt('lambda_term_energyFluid', lambda_term_energyFluid)
np.savetxt('lambda_term_energyWall', lambda_term_energyWall)
np.savetxt('beta_term_densityFluid', beta_term_densityFluid)
np.savetxt('beta_term_energyFluid', beta_term_energyFluid)
np.savetxt('beta_term_energyWall', beta_term_energyWall)
np.savetxt('betaWall_term_densityFluid', betaWall_term_densityFluid)
np.savetxt('betaWall_term_energyFluid', betaWall_term_energyFluid)
np.savetxt('betaWall_term_energyWall', betaWall_term_energyWall)

# Save conjugate variables (mu, beta, lambdaUpper, lambdaLower, betaUpper, betaLower)
np.savetxt('lambda', Lambda)
np.savetxt('beta', beta.T)
np.savetxt('betaWall', betaWall.T)

# Move the outputs to a new folder named "conjugate_variables"
#shutil.rmtree('./conjugate_variables')
conjugate_variables = r'./conjugate_variables'
if not os.path.exists(conjugate_variables):
    os.makedirs(conjugate_variables)
for f in os.listdir('./'):
    if (f.startswith('mu') or f.startswith('beta') or f.startswith('lambda')):
        shutil.move(f, conjugate_variables)
#EOF
