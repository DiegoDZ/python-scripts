##############################################
#COMPUTES THE DISSIPATIVE MATRIX D(t)
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
Ct = np.loadtxt('Ctmatrix_rhoeg')

##############################################
#DEFINE VARIABLES
##############################################
number_correlations_files = 9
steps = len(Ct)
number_nodes = int(np.sqrt(len(Ct[0]) / number_correlations_files ))
M = np.zeros((3 * number_nodes, 3 * number_nodes))
D = np.zeros((steps, number_nodes ** 2 * number_correlations_files))
Ctdev = np.zeros((steps-1, number_nodes ** 2 * number_correlations_files))
Ct2dev = np.zeros((steps-1, number_nodes ** 2 * number_correlations_files))
Ctinv = np.zeros((steps-1, number_nodes ** 2 * number_correlations_files))
Ctnorm = np.zeros((steps-1, number_nodes ** 2 * number_correlations_files))
CtdevCtinv = np.zeros((steps-1, number_nodes ** 2 * number_correlations_files)) #-1 porque no calculo estas variables en el ultimo paso de tiempo
#Box length
Lx = 17.3162
Ly = 17.3162
Lz = 34.6325
#Bin Size
dz = Lz / number_nodes
#Bin volume
V = dz * Lx * Ly
#Time step
dt = 0.005

#####################################################################################################################################
############################################## COMPUTES FOR t = 0 ###################################################################
#####################################################################################################################################

##############################################
#COMPUTE C(t=0), R(as Cinv(t=0)) and C(t) normalized (as C(t)*R)at t = 0
##############################################
#Select the block elements of the matrix C(t=0)
Ct0_00 = np.asmatrix(Ct[0, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_01 = np.asmatrix(Ct[0, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_02 = np.asmatrix(Ct[0, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_10 = np.asmatrix(Ct[0, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_11 = np.asmatrix(Ct[0, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_12 = np.asmatrix(Ct[0, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_20 = np.asmatrix(Ct[0, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_21 = np.asmatrix(Ct[0, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct0_22 = np.asmatrix(Ct[0, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Create the matrix C(t=0)
Ct0 = np.bmat(([Ct0_00, Ct0_01, Ct0_02],[Ct0_10, Ct0_11, Ct0_12],[Ct0_20, Ct0_21, Ct0_22]))
Ct0_stat = (Ct0 + Ct0.T) /2
#Calculate the matrix R as the inverse of the matrix C(t=0)
R = linalg.pinv(Ct0, rcond = 1e-12)
#Change the format of R in order to obtain the inverse of C in each time step. Matrix-> Vector
Ctinv0 = R
Ctinv[0,:] = np.bmat((Ctinv0[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctinv0[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctinv0[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Ctinv0[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctinv0[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctinv0[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , Ctinv0[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctinv0[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctinv0[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
#Calculate C(t) normalized
Ctnorm0 = Ct0_stat.dot(R)
#Change format: Matrix -> Vector
Ctnorm[0,:] = np.bmat((Ctnorm0[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctnorm0[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctnorm0[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Ctnorm0[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctnorm0[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctnorm0[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , Ctnorm0[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctnorm0[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctnorm0[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))


##############################################
#COMPUTE L as Cdev(t=0)
##############################################
#Select the block elements of the matrix C(t=1)
Ct1_00 = np.asmatrix(Ct[1, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_01 = np.asmatrix(Ct[1, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_02 = np.asmatrix(Ct[1, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_10 = np.asmatrix(Ct[1, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_11 = np.asmatrix(Ct[1, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_12 = np.asmatrix(Ct[1, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_20 = np.asmatrix(Ct[1, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_21 = np.asmatrix(Ct[1, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
Ct1_22 = np.asmatrix(Ct[1, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Create the matrix C(t=1)
Ct1 = np.bmat(([Ct1_00, Ct1_01, Ct1_02],[Ct1_10, Ct1_11, Ct1_12],[Ct1_20, Ct1_21, Ct1_22]))
#Ct1_stat = (Ct1 + Ct1.T) /2
L = (Ct1 - Ct1.T) / (2 * dt)
L_stat = (L - L.T) / 2
#Change the format of L in order to obtain the derivative of C in each tiem step
Ctdev0 = L_stat
Ctdev[0,:] = np.bmat((Ctdev0[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctdev0[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctdev0[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Ctdev0[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctdev0[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctdev0[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , Ctdev0[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ctdev0[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ctdev0[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))

#####################################
#COMPUTE THE SECOND DERIVATIVE OF C AT t=0
#####################################
Ct2dev0 = (Ct1 + Ct1.T - 2*Ct0_stat) / dt **2
#Change format: Matrix -> Vector
Ct2dev[0,:] = np.bmat((Ct2dev0[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ct2dev0[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ct2dev0[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Ct2dev0[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ct2dev0[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ct2dev0[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , Ct2dev0[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Ct2dev0[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Ct2dev0[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))

#####################################
#CALCULATE D at t=0 as -2*L
#####################################
D0 = - 2 * L
#Change format: Matrix -> Vector
D[0,:] = np.bmat((D0[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), D0[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), D0[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , D0[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), D0[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), D0[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , D0[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), D0[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), D0[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))

#####################################################################################################################################
############################################## COMPUTE FOR t >0 ###################################################################
#####################################################################################################################################

for t in range(1, steps-1, 1):
#############
#D(t)
#############
    #Select the blocks of the matrix C(t)
    C_00 = np.asmatrix(Ct[t, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_01 = np.asmatrix(Ct[t, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_02 = np.asmatrix(Ct[t, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_10 = np.asmatrix(Ct[t, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_11 = np.asmatrix(Ct[t, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_12 = np.asmatrix(Ct[t, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_20 = np.asmatrix(Ct[t, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_21 = np.asmatrix(Ct[t, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    C_22 = np.asmatrix(Ct[t, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    #Create the matrix C at time t
    C = np.bmat(([C_00, C_01, C_02],[C_10, C_11, C_12],[C_20, C_21, C_22]))
    #Calculate the inverse of C
    Cinv = linalg.pinv(C, rcond = 1e-12)
    #Select the blocks of the matrix C(t+dt)
    Cforward_00 = np.asmatrix(Ct[t+1, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_01 = np.asmatrix(Ct[t+1, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_02 = np.asmatrix(Ct[t+1, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_10 = np.asmatrix(Ct[t+1, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_11 = np.asmatrix(Ct[t+1, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_12 = np.asmatrix(Ct[t+1, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_20 = np.asmatrix(Ct[t+1, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_21 = np.asmatrix(Ct[t+1, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cforward_22 = np.asmatrix(Ct[t+1, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    #Create the matrix C at time t+dt
    Cforward = np.bmat(([Cforward_00, Cforward_01, Cforward_02],[Cforward_10, Cforward_11, Cforward_12],[Cforward_20, Cforward_21, Cforward_22]))
    #Select the blocks of the matrix C(t-dt)
    Cbackward_00 = np.asmatrix(Ct[t-1, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_01 = np.asmatrix(Ct[t-1, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_02 = np.asmatrix(Ct[t-1, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_10 = np.asmatrix(Ct[t-1, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_11 = np.asmatrix(Ct[t-1, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_12 = np.asmatrix(Ct[t-1, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_20 = np.asmatrix(Ct[t-1, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_21 = np.asmatrix(Ct[t-1, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    Cbackward_22 = np.asmatrix(Ct[t-1, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
    #Create the matrix C at time t-dt
    Cbackward = np.bmat(([Cbackward_00, Cbackward_01, Cbackward_02],[Cbackward_10, Cbackward_11, Cbackward_12],[Cbackward_20, Cbackward_21, Cbackward_22]))
    #Derive C at time t
    Cdev = (Cforward - Cbackward) / (2 * dt)
    #Compute D matrix as -(Cdev(t) + L*R*C(t)) * Cinv(t)*Rinv
    M = -np.asmatrix((Cdev + L.dot(R).dot(C)).dot(Cinv).dot(Ct0_stat))
    #Save it in matrix format
    D[t,:] = np.bmat((M[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), M[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), M[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
            , M[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), M[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), M[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , M[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), M[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), M[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
    #Calculate the 2nd derivative of C
    C2dev =(Cforward + Cbackward - 2 * C) / dt**2
    #Save the matrix C2dev at each time step
    Ct2dev[t,:] = np.bmat((C2dev[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), C2dev[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), C2dev[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
            , C2dev[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), C2dev[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), C2dev[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , C2dev[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), C2dev[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), C2dev[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
################
#C(t) normalized
################
    #Calculate C normalized and change format: Matrix -> Vector
    Cnorm = C.dot(R)
    Ctnorm[t,:] = np.bmat((Cnorm[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cnorm[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cnorm[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                    , Cnorm[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cnorm[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cnorm[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                        , Cnorm[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cnorm[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cnorm[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
################
#Cdev(t)*Cinv(t)
################
    #Compute Cinv
    Cinv = linalg.pinv(C, rcond = 1e-12)
    #Compute the product Cdev(t)*Cinv(t)
    P = Cdev.dot(Cinv)
    #Save the product at each time step
    CtdevCtinv[t,:] = np.bmat((P[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), P[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), P[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
            , P[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), P[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), P[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , P[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), P[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), P[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
    #Save the matrix Cinv at each time step
    Ctinv[t,:] = np.bmat((Cinv[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cinv[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cinv[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
            , Cinv[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cinv[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cinv[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Cinv[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cinv[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cinv[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
    #Save the matrix Cdev at each time step
    Ctdev[t,:] = np.bmat((Cdev[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cdev[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cdev[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
            , Cdev[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cdev[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cdev[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
                , Cdev[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cdev[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cdev[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))

#####################################################################################################################################
############################################## COMPUTE FOR LAST TIME STEP ###########################################################
#####################################################################################################################################

#######################################
#CALCULATE THE LAST TIME STEP OF Ctdev AND Ctinv
########################################

##Ctdev
##Copy the penultimate row of Ctdev and paste it in the last row of Ctdev
#Ctdev[steps-1,:] = Ctdev[steps-2,:]
#
##Ctinv
##Select the blocks of the matrix C(t)
#Cstep_00 = np.asmatrix(Ct[steps-1, 0:len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_01 = np.asmatrix(Ct[steps-1, len(Ct[0])/number_correlations_files:2 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_02 = np.asmatrix(Ct[steps-1, 2*len(Ct[0])/number_correlations_files:3 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_10 = np.asmatrix(Ct[steps-1, 3*len(Ct[0])/number_correlations_files:4 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_11 = np.asmatrix(Ct[steps-1, 4*len(Ct[0])/number_correlations_files:5 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_12 = np.asmatrix(Ct[steps-1, 5*len(Ct[0])/number_correlations_files:6 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_20 = np.asmatrix(Ct[steps-1, 6*len(Ct[0])/number_correlations_files:7 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_21 = np.asmatrix(Ct[steps-1, 7*len(Ct[0])/number_correlations_files:8 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
#Cstep_22 = np.asmatrix(Ct[steps-1, 8*len(Ct[0])/number_correlations_files:9 * len(Ct[0])/number_correlations_files].reshape(number_nodes, number_nodes))
##Create the matrix C at last time step
#Cstep = np.bmat(([Cstep_00, Cstep_01, Cstep_02],[Cstep_10, Cstep_11, Cstep_12],[Cstep_20, Cstep_21, Cstep_22]))
##Calculate the inverse of C at last time step
#Cstepinv = linalg.pinv(Cstep, rcond = 1e-12)
##Change the format: Matrix -> Vector
#Cstep = np.bmat((Cstep[0:number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cstep[0:number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cstep[0:number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
#          , Cstep[number_nodes:2*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cstep[number_nodes:2*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cstep[number_nodes:2*number_nodes,2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)\
#          , Cstep[2*number_nodes:3*number_nodes, 0:number_nodes].reshape(1,number_nodes**2), Cstep[2*number_nodes:3*number_nodes, number_nodes:2*number_nodes].reshape(1,number_nodes**2), Cstep[2*number_nodes:3*number_nodes, 2*number_nodes:3*number_nodes].reshape(1,number_nodes**2)))
##Copy Cstep to the last row of Ctinv
#Ctinv[steps-1,:] = Cstep


#####################################################################################################################################
############################################ COMPUTE THE INTEGRAL OF D ##############################################################
#####################################################################################################################################
M7_integral = np.sum(D[0:7,:], axis =0) * V * dt
D7_integral = np.bmat(([M7_integral[0:10000].reshape(number_nodes,number_nodes), M7_integral[10000:20000].reshape(number_nodes,number_nodes), M7_integral[20000:30000].reshape(number_nodes,number_nodes)],[M7_integral[30000:40000].reshape(number_nodes,number_nodes), M7_integral[40000:50000].reshape(number_nodes,number_nodes), M7_integral[50000:60000].reshape(number_nodes,number_nodes)], [M7_integral[60000:70000].reshape(number_nodes,number_nodes), M7_integral[70000:80000].reshape(number_nodes,number_nodes), M7_integral[80000:90000].reshape(number_nodes,number_nodes)]))
#M7 = D[7,:]
#D7 = np.bmat(([M7[0:10000].reshape(number_nodes,number_nodes), M7[10000:20000].reshape(number_nodes,number_nodes), M7[20000:30000].reshape(number_nodes,number_nodes)],[M7[30000:40000].reshape(number_nodes,number_nodes), M7[40000:50000].reshape(number_nodes,number_nodes), M7[50000:60000].reshape(number_nodes,number_nodes)], [M7[60000:70000].reshape(number_nodes,number_nodes), M7[70000:80000].reshape(number_nodes,number_nodes), M7[80000:90000].reshape(number_nodes,number_nodes)]))

#####################################################################################################################################
###################################################### SAVE THE OUTPUT ##############################################################
#####################################################################################################################################
np.savetxt('D_rhoeg', D)
np.savetxt('D7_integral_rhoeg', D7_integral)
np.savetxt('Ct0_rhoeg', Ct0_stat)
np.savetxt('L_rhoeg',L_stat)
np.savetxt('R_rhoeg', R)
np.savetxt('Ctnorm', Ctnorm)
np.savetxt('Ctinv', Ctinv)
np.savetxt('Ctdev', Ctdev)
np.savetxt('Ct2dev', Ct2dev)
np.savetxt('CtdevCtinv', CtdevCtinv)

#EOF
