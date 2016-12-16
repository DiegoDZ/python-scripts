# -*- coding: utf-8 -*-
"""
Author: @DiegoDZ

Date: 2 dec 2016

"""

"""
This script takes the correlation files and averages them.

Initial parameters: 	number_simulations

run: python average-correlations.py

"""

import os
import numpy as np

#Define parameters
number_simulations = 100
number_nodes = len((np.loadtxt('corr_QQ.1'))[0])
number_snapshots = len(np.loadtxt('corr_QQ.1'))

#Create arrays for outputs
corr_QQ = np.zeros((number_snapshots, number_nodes))
corr_QPi = np.zeros((number_snapshots, number_nodes))
corr_PiQ = np.zeros((number_snapshots, number_nodes))
corr_PiPi = np.zeros((number_snapshots, number_nodes))
corr_iLeiLe = np.zeros((number_snapshots, number_nodes))

#Average
for file_name in os.listdir('./'):
    if file_name[0:8] == "corr_QQ.":
        corr_QQ += np.loadtxt(file_name) / number_simulations
    elif file_name[0:9] == "corr_QPi.":
        corr_QPi += np.loadtxt(file_name) / number_simulations
    elif file_name[0:9] == "corr_PiQ.":
        corr_PiQ += np.loadtxt(file_name) / number_simulations
    elif file_name[0:10] == "corr_PiPi.":
        corr_PiPi += np.loadtxt(file_name) / number_simulations
    elif file_name[0:12] == "corr_iLeiLe.":
        corr_iLeiLe += np.loadtxt(file_name) / number_simulations

#Save outputs
np.savetxt('corr_QQ', corr_QQ)
np.savetxt('corr_QPi', corr_QPi)
np.savetxt('corr_PiQ', corr_PiQ)
np.savetxt('corr_PiPi', corr_PiPi)
np.savetxt('corr_iLeiLe', corr_iLeiLe)

#EOF
