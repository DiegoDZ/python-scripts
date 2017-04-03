##############################################
# THIS SCRIPT COMPUTES THE DISSIPATIVE MATRIX M(t)
#
# AUTHOR: @DiegoDZ
# DATE: MARCH 2017
#
# run: >>> python Dmatrix_rhoegTheory.py
##############################################

import numpy as np
from scipy import linalg

##############################################
#LOAD FILES
##############################################
Ct = np.loadtxt('Ctmatrix_rhoegStat')

##############################################
#DEFINE VARIABLES
##############################################
blocks = 9
tol = 1e-3 #rcond in linalg.pinv
steps = len(Ct)
nodes = int(np.sqrt(len(Ct[0]) / blocks ))
Mt = np.zeros((steps, nodes ** 2 * blocks))
Ctdev = np.zeros((steps-1, nodes ** 2 * blocks)) #-1 because the script does not compute the values for the last time step
Ctinv = np.zeros((steps-1, nodes ** 2 * blocks))
Ctnorm = np.zeros((steps-1, nodes ** 2 * blocks))
Ctnorminv = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtnorminv = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtnorminvPlusL = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtinv = np.zeros((steps-1, nodes ** 2 * blocks))
#########Simulation details########
#Box length
Lx = 17.3162
Ly = 17.3162
Lz = 34.6325
#Bin Size
dz = Lz / nodes
#Bin volume
V = dz * Lx * Ly
#Time step
dt = 0.005

##############################################
#CREATE THE FUNCTION reshape_mv (MATRIX -> VECTOR)
##############################################

def reshape_mv(A):
    B = np.bmat((A[0:nodes, 0:nodes].reshape(1,nodes**2), A[0:nodes, nodes:2*nodes].reshape(1,nodes**2), A[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
    ,A[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), A[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), A[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
    ,A[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), A[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), A[2*nodes:3*nodes,2*nodes:3*nodes].reshape(1,nodes**2)))
    return B



############################################## COMPUTES FOR t = 0 ###################################################################

##############################################
#COMPUTE C(t=0), R(as Cinv(t=0)) and C(t) normalized (as C(t)*R) at t = 0
##############################################
#########Create the matrix C(t=0)########
Ct0_00 = np.asmatrix(Ct[0, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_01 = np.asmatrix(Ct[0, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_02 = np.asmatrix(Ct[0, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_10 = np.asmatrix(Ct[0, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_11 = np.asmatrix(Ct[0, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_12 = np.asmatrix(Ct[0, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_20 = np.asmatrix(Ct[0, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_21 = np.asmatrix(Ct[0, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_22 = np.asmatrix(Ct[0, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0 = np.bmat(([Ct0_00, Ct0_01, Ct0_02],[Ct0_10, Ct0_11, Ct0_12],[Ct0_20, Ct0_21, Ct0_22]))
Ct0_stat = (Ct0 + Ct0.T) / 2 #Increase the statistic becasuse C(t=0) is symmetrical
#########Compute R and Ctinv(t=0)########
R = linalg.pinv(Ct0_stat, rcond = tol)
Ctinv0 = R
Ctinv[0,:] = reshape_mv(Ctinv0)
#########Compute C(t=0) normalized and its inverse########
Ctnorm0 = Ct0_stat.dot(R)
Ctnorminv0 = linalg.pinv(Ctnorm0, rcond = tol)
Ctnorm[0,:] = reshape_mv(Ctnorm0)
Ctnorminv[0,:] = reshape_mv(Ctnorminv0)

##############################################
#COMPUTE L as -Cdev(t=0)
##############################################
#Select the block elements of the matrix C(t=1)
Ct1_00 = np.asmatrix(Ct[1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_01 = np.asmatrix(Ct[1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_02 = np.asmatrix(Ct[1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_10 = np.asmatrix(Ct[1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_11 = np.asmatrix(Ct[1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_12 = np.asmatrix(Ct[1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_20 = np.asmatrix(Ct[1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_21 = np.asmatrix(Ct[1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1_22 = np.asmatrix(Ct[1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct1 = np.bmat(([Ct1_00, Ct1_01, Ct1_02],[Ct1_10, Ct1_11, Ct1_12],[Ct1_20, Ct1_21, Ct1_22]))
L = - (Ct1 - Ct1.T) / (2 * dt)
L_antiSym = (L - L.T) / 2 #Increase the statistic becasuse L is antisymmetric

#More statistic using the time reversal propierty
e00 = np.asmatrix(np.identity(nodes))
e01 = np.asmatrix(np.zeros((nodes,nodes)))
e02 = np.asmatrix(np.zeros((nodes,nodes)))
e10 = np.asmatrix(np.zeros((nodes,nodes)))
e11 = np.asmatrix(np.identity(nodes))
e12 = np.asmatrix(np.zeros((nodes,nodes)))
e20 = np.asmatrix(np.zeros((nodes,nodes)))
e21 = np.asmatrix(np.zeros((nodes,nodes)))
e22 = np.asmatrix(-np.identity(nodes))
epsilon = np.bmat(([e00,e01,e02],[e10,e11,e12],[e20,e21,e22]))
L_stat = (L - epsilon.dot(L_antiSym).dot(epsilon)) / 2
Ctdev0 = - L_stat
Ctdev[0,:] = reshape_mv(Ctdev0)

#######################################
#COMPUTE Ctdev*Ctnorm^-1 AT t=0
#######################################
CtdevCtnorminv0 = Ctdev0.dot(Ctnorminv0)
CtdevCtnorminv[0,:] = reshape_mv(CtdevCtnorminv0)

#######################################
#COMPUTE Ctdev*Ctnorm^-1+L AT t=0
#######################################
CtdevCtnorminvPlusL0 = CtdevCtnorminv0 + L_stat
CtdevCtnorminvPlusL[0,:] = reshape_mv(CtdevCtnorminvPlusL0)

#####################################
#COMPUTE CtdevCt  AT t=0
#####################################
CtdevCtinv0 = Ctdev0.dot(R)
CtdevCtinv[0,:] = reshape_mv(CtdevCtinv0)

#####################################
#CALCULATE M at t=0 as -2*L
#####################################
M0 = - 2 * L
Mt[0,:] = reshape_mv(M0)



############################################## COMPUTE FOR t >0 ###################################################################

for t in range(1, steps-1, 1):
    #####################################
    # COMPUTE Cinv
    #####################################
        #Create the matrix C(t) and its inverse
        C_00 = np.asmatrix(Ct[t, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
        C_01 = np.asmatrix(Ct[t, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_02 = np.asmatrix(Ct[t, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_10 = np.asmatrix(Ct[t, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_11 = np.asmatrix(Ct[t, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_12 = np.asmatrix(Ct[t, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_20 = np.asmatrix(Ct[t, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_21 = np.asmatrix(Ct[t, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C_22 = np.asmatrix(Ct[t, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
        C = np.bmat(([C_00, C_01, C_02],[C_10, C_11, C_12],[C_20, C_21, C_22]))
        Cinv = linalg.pinv(C, rcond = tol)

    #####################################
    # COMPUTE C(t) NORMALIZED AND ITS INVERSE
    #####################################
        Cnorm = R.dot(C)
        Cnorminv = linalg.pinv(Cnorm, rcond = tol)
        Ctnorm[t,:] = reshape_mv(Cnorm)
        Ctnorminv[t,:] = reshape_mv(Cnorminv)

    #####################################
    # DERIVE C(t)
    #####################################
        #Create the matrix C(t+dt) and its inverse
        Cforward_00 = np.asmatrix(Ct[t+1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_01 = np.asmatrix(Ct[t+1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_02 = np.asmatrix(Ct[t+1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_10 = np.asmatrix(Ct[t+1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_11 = np.asmatrix(Ct[t+1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_12 = np.asmatrix(Ct[t+1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_20 = np.asmatrix(Ct[t+1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_21 = np.asmatrix(Ct[t+1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward_22 = np.asmatrix(Ct[t+1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cforward = np.bmat(([Cforward_00, Cforward_01, Cforward_02],[Cforward_10, Cforward_11, Cforward_12],[Cforward_20, Cforward_21, Cforward_22]))
        #Create the matrix C(t-dt)
        Cbackward_00 = np.asmatrix(Ct[t-1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_01 = np.asmatrix(Ct[t-1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_02 = np.asmatrix(Ct[t-1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_10 = np.asmatrix(Ct[t-1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_11 = np.asmatrix(Ct[t-1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_12 = np.asmatrix(Ct[t-1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_20 = np.asmatrix(Ct[t-1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_21 = np.asmatrix(Ct[t-1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward_22 = np.asmatrix(Ct[t-1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
        Cbackward = np.bmat(([Cbackward_00, Cbackward_01, Cbackward_02],[Cbackward_10, Cbackward_11, Cbackward_12],[Cbackward_20, Cbackward_21, Cbackward_22]))
        #Derive C at time t
        Cdev = (Cforward - Cbackward) / (2 * dt)
        Ctdev[t,:] = reshape_mv(Cdev)

    #####################################
    # COMPUTE M(t)
    #####################################
        M = -np.asmatrix((Cdev.dot(Cnorm) - Ctdev0))
        Mt[t,:] = reshape_mv(M)

    #####################################
    # COMPUTE Cdev(t) * Cnorm(t) - L
    #####################################
        CdevCnorminvPlusL = Cdev.dot(Cnorminv) - L_stat
        CtdevCtnorminvPlusL[t,:] = reshape_mv(CdevCnorminvPlusL)

    #####################################
    # COMPUTE Cdev(t) * Cnorminv(t)
    #####################################
        Cnorminv = linalg.pinv(Cnorm, rcond =tol)
        Ctnorminv[t,:] = reshape_mv(Cnorminv)
        P = Cdev.dot(Cnorminv)
        CtdevCtnorminv[t,:] = reshape_mv(P)


############################################ COMPUTE THE INTEGRAL OF D ##############################################################

#M7_integral = np.sum(Mt[0:7,:], axis =0) * V * dt
#M7_integral = np.bmat(([M7_integral[0:nodes**2].reshape(nodes,nodes), M7_integral[nodes**2:2*nodes**2].reshape(nodes,nodes), M7_integral[2*nodes**2:3*nodes**2].reshape(nodes,nodes)],[M7_integral[3*nodes**2:4*nodes**2].reshape(nodes,nodes), M7_integral[4*nodes**2:5*nodes**2].reshape(nodes,nodes), M7_integral[5*nodes**2:6*nodes**2].reshape(nodes,nodes)], [M7_integral[6*nodes**2:7*nodes**2].reshape(nodes,nodes), M7_integral[7*nodes**2:8*nodes**2].reshape(nodes,nodes), M7_integral[8*nodes**2:9*nodes**2].reshape(nodes,nodes)]))


###################################################### SAVE THE OUTPUT ##############################################################

np.savetxt('Mt_rhoeg', Mt)
#np.savetxt('M7_integral_rhoeg', M7_integral)
np.savetxt('Ct0_rhoeg', Ct0_stat)
np.savetxt('L_rhoeg',L_stat)
np.savetxt('R_rhoeg', R)
np.savetxt('Ctnorminv', Ctnorminv)
np.savetxt('Ctnorm', Ctnorm)
np.savetxt('Ctinv', Ctinv)
np.savetxt('Ctdev', Ctdev)
np.savetxt('CtdevCtnorminv', CtdevCtnorminv)
np.savetxt('CtdevCtnorminvPlusL', CtdevCtnorminvPlusL)

#EOF
