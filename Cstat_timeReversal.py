##########################################################################################
# This script increases the statistic of the matrix C(t) using the time reversal propierty
# Author: @DiegoDZ
# Date: March 2017
# run: >>> Cstat_timereversal.py
##########################################################################################

import numpy as np

#Load C(t) matrix
Ct = np.loadtxt('Ctmatrix_rhoeg')
#Define variables
blocks = 9
steps = len(Ct)
nodes = int(np.sqrt(len(Ct[0]) / blocks ))
Ct_rhoegStat = np.zeros((steps, nodes ** 2 * blocks))

#Create the matrix epsilon
E00 = np.asmatrix(np.identity(nodes))
E01 = np.asmatrix(np.zeros((nodes,nodes)))
E02 = np.asmatrix(np.zeros((nodes,nodes)))
E10 = np.asmatrix(np.zeros((nodes,nodes)))
E11 = np.asmatrix(np.identity(nodes))
E12 = np.asmatrix(np.zeros((nodes,nodes)))
E20 = np.asmatrix(np.zeros((nodes,nodes)))
E21 = np.asmatrix(np.zeros((nodes,nodes)))
E22 = np.asmatrix(-np.identity(nodes))
epsilon = np.bmat(([E00,E01,E02],[E10,E11,E12],[E20,E21,E22]))

#Compute for each step the C(t) matrix with more statistic
for t in range(0, steps, 1):
    #Select the blocks of the matrix C(t)
    C_00 = np.asmatrix(Ct[t, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
    C_01 = np.asmatrix(Ct[t, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_02 = np.asmatrix(Ct[t, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_10 = np.asmatrix(Ct[t, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_11 = np.asmatrix(Ct[t, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_12 = np.asmatrix(Ct[t, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_20 = np.asmatrix(Ct[t, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_21 = np.asmatrix(Ct[t, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
    C_22 = np.asmatrix(Ct[t, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
    #Create the matrix C at time t
    C = np.bmat(([C_00, C_01, C_02],[C_10, C_11, C_12],[C_20, C_21, C_22]))
    #Computes (C + epsilon*C.T*epsilon) / 2
    C_stat = (C + epsilon.dot(C.T).dot(epsilon)) / 2
    #Change format: Matrix -> Vector
    Ct_rhoegStat[t,:] = np.bmat((C_stat[0:nodes, 0:nodes].reshape(1,nodes**2), C_stat[0:nodes, nodes:2*nodes].reshape(1,nodes**2), C_stat[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                    , C_stat[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), C_stat[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), C_stat[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                        , C_stat[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), C_stat[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), C_stat[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#Save the output
np.savetxt('Ctmatrix_rhoegStat', Ct_rhoegStat)

#EOF
