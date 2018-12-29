###################################
# This script computes the Laplace transform of the matrix of correlations C(t) and the sim of the matrices L+M
#
# Author: DiegoDZ
# Date: June 2017
#
###################################

import numpy as np
from scipy import linalg
import datetime

print datetime.datetime.now()
##################################################
#LOAD FILE AND DEFINE VARIABLES
##################################################
Ct = np.loadtxt('Ctmatrix_rhoegStat')
blocks = 9
dt = 1
nRows, nCols = np.shape(Ct)
nodes = int(np.sqrt(nCols / blocks ))
time = np.array(range(0, nRows, 1))
s = np.array((0, 0.01, 0.0025, 0.0014, 0.0005))

##################################################
#CREATE C(t=0)
##################################################
C0_00 = np.asmatrix(Ct[0, 0 * nCols/blocks: 1 * nCols/blocks].reshape(nodes, nodes))
C0_01 = np.asmatrix(Ct[0, 1 * nCols/blocks: 2 * nCols/blocks].reshape(nodes, nodes))
C0_02 = np.asmatrix(Ct[0, 2 * nCols/blocks: 3 * nCols/blocks].reshape(nodes, nodes))
C0_10 = np.asmatrix(Ct[0, 3 * nCols/blocks: 4 * nCols/blocks].reshape(nodes, nodes))
C0_11 = np.asmatrix(Ct[0, 4 * nCols/blocks: 5 * nCols/blocks].reshape(nodes, nodes))
C0_12 = np.asmatrix(Ct[0, 5 * nCols/blocks: 6 * nCols/blocks].reshape(nodes, nodes))
C0_20 = np.asmatrix(Ct[0, 6 * nCols/blocks: 7 * nCols/blocks].reshape(nodes, nodes))
C0_21 = np.asmatrix(Ct[0, 7 * nCols/blocks: 8 * nCols/blocks].reshape(nodes, nodes))
C0_22 = np.asmatrix(Ct[0, 8 * nCols/blocks: 9 * nCols/blocks].reshape(nodes, nodes))
C0 = np.bmat(([C0_00, C0_01, C0_02],[C0_10, C0_11, C0_12],[C0_20, C0_21, C0_22]))
C0_stat = (C0 + C0.T) / 2 #C(t=0) is symmetric

##################################################
#COMPUTE LAPLACE TRANSFORM AND L + M
##################################################
for i in range(len(s)):
    # Calculate the integral
    Clap = np.sum((np.exp(-s[i] * time[:, np.newaxis]) * Ct) * dt, axis = 0)
    # Change format (From vector to matrix)
    Clap_00 = np.asmatrix(Clap[0 * nCols/blocks: 1 * nCols/blocks].reshape(nodes, nodes))
    Clap_01 = np.asmatrix(Clap[1 * nCols/blocks: 2 * nCols/blocks].reshape(nodes, nodes))
    Clap_02 = np.asmatrix(Clap[2 * nCols/blocks: 3 * nCols/blocks].reshape(nodes, nodes))
    Clap_10 = np.asmatrix(Clap[3 * nCols/blocks: 4 * nCols/blocks].reshape(nodes, nodes))
    Clap_11 = np.asmatrix(Clap[4 * nCols/blocks: 5 * nCols/blocks].reshape(nodes, nodes))
    Clap_12 = np.asmatrix(Clap[5 * nCols/blocks: 6 * nCols/blocks].reshape(nodes, nodes))
    Clap_20 = np.asmatrix(Clap[6 * nCols/blocks: 7 * nCols/blocks].reshape(nodes, nodes))
    Clap_21 = np.asmatrix(Clap[7 * nCols/blocks: 8 * nCols/blocks].reshape(nodes, nodes))
    Clap_22 = np.asmatrix(Clap[8 * nCols/blocks: 9 * nCols/blocks].reshape(nodes, nodes))
    Claplace = np.bmat(([Clap_00, Clap_01, Clap_02],[Clap_10, Clap_11, Clap_12],[Clap_20, Clap_21, Clap_22]))
    # Calculate L + M
    LplusM = C0_stat * linalg.pinv(Claplace) * C0_stat - s[i] * C0_stat
    # Save the results
    np.savetxt('Claplace' + str(i), Claplace)
    np.savetxt('LplusM' + str(i), LplusM)

print datetime.datetime.now()
#EOF

