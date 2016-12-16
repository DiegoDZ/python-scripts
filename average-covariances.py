# -*- coding: utf-8 -*-
"""
Author: @DiegoDZ

Date: 2 dec 2016

"""

"""
This script takes the covariance files and averages them.

Initial parameters: 	number_simulations

run: python average-covariances.py

"""

import os
import numpy as np

#Define parameters
number_simulations = 100
number_nodes = len(np.loadtxt('cov_ee.1'))

#Create arrays for outputs
cov_eQ = np.zeros((number_nodes, number_nodes))
cov_ePi = np.zeros((number_nodes, number_nodes))
cov_ee = np.zeros((number_nodes, number_nodes))
cov_eiLe = np.zeros((number_nodes, number_nodes))

#Average
for file_name in os.listdir('./'):
    if file_name[0:7] == "cov_eQ.":
        cov_eQ += np.loadtxt(file_name) / number_simulations
    elif file_name[0:8] == "cov_ePi.":
        cov_ePi += np.loadtxt(file_name) / number_simulations
    elif file_name[0:7] == "cov_ee.":
        cov_ee += np.loadtxt(file_name) / number_simulations
    elif file_name[0:9] == "cov_eiLe.":
        cov_eiLe += np.loadtxt(file_name) / number_simulations

#Save outputs
np.savetxt('cov_eQ', cov_eQ)
np.savetxt('cov_ePi', cov_ePi)
np.savetxt('cov_ee', cov_ee)
np.savetxt('cov_eiLe', cov_eiLe)

#EOF
