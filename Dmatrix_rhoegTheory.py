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
Ct = np.loadtxt('Ctmatrix_rhoegStat')

##############################################
#DEFINE VARIABLES
##############################################
blocks = 9
tol = 1e-3 #rcond in linalg.pinv
steps = len(Ct)
nodes = int(np.sqrt(len(Ct[0]) / blocks ))
M = np.zeros((3 * nodes, 3 * nodes))
D = np.zeros((steps, nodes ** 2 * blocks))
Ctdev = np.zeros((steps-1, nodes ** 2 * blocks))
#Ct2dev = np.zeros((steps-1, nodes ** 2 * blocks))
Ctinv = np.zeros((steps-1, nodes ** 2 * blocks))
Ctnorm = np.zeros((steps-1, nodes ** 2 * blocks))
Ctnorminv = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtnorminv = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtnorminvMinusL = np.zeros((steps-1, nodes ** 2 * blocks))
CtdevCtinv = np.zeros((steps-1, nodes ** 2 * blocks)) #-1 porque no calculo estas variables en el ultimo paso de tiempo
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


#####################################################################################################################################
############################################## COMPUTES FOR t = 0 ###################################################################
#####################################################################################################################################

##############################################
#COMPUTE C(t=0), R(as Cinv(t=0)) and C(t) normalized (as C(t)*R)at t = 0
##############################################
#Select the block elements of the matrix C(t=0)
Ct0_00 = np.asmatrix(Ct[0, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_01 = np.asmatrix(Ct[0, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_02 = np.asmatrix(Ct[0, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_10 = np.asmatrix(Ct[0, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_11 = np.asmatrix(Ct[0, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_12 = np.asmatrix(Ct[0, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_20 = np.asmatrix(Ct[0, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_21 = np.asmatrix(Ct[0, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
Ct0_22 = np.asmatrix(Ct[0, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Create the matrix C(t=0)
Ct0 = np.bmat(([Ct0_00, Ct0_01, Ct0_02],[Ct0_10, Ct0_11, Ct0_12],[Ct0_20, Ct0_21, Ct0_22]))
Ct0_stat = (Ct0 + Ct0.T) / 2
#Calculate the matrix R as the inverse of the matrix C(t=0)
R = linalg.pinv(Ct0_stat, rcond = tol)
#Change the format of R in order to obtain the inverse of C in each time step. Matrix-> Vector
Ctinv0 = R
Ctinv[0,:] = np.bmat((Ctinv0[0:nodes, 0:nodes].reshape(1,nodes**2), Ctinv0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Ctinv0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , Ctinv0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Ctinv0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctinv0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Ctinv0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Ctinv0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctinv0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
#Calculate C(t) normalized and its inverse
Ctnorm0 = Ct0_stat.dot(R)
Ctnorminv0 = linalg.pinv(Ctnorm0, rcond = tol)
#Change format: Matrix -> Vector
Ctnorm[0,:] = np.bmat((Ctnorm0[0:nodes, 0:nodes].reshape(1,nodes**2), Ctnorm0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorm0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , Ctnorm0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Ctnorm0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorm0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Ctnorm0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Ctnorm0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorm0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
#Change format: Matrix-> Vector
Ctnorminv[0,:] = np.bmat((Ctnorminv0[0:nodes, 0:nodes].reshape(1,nodes**2), Ctnorminv0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorminv0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , Ctnorminv0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Ctnorminv0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorminv0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Ctnorminv0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Ctnorminv0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctnorminv0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

##############################################
#COMPUTE L as Cdev(t=0)
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
#Create the matrix C(t=1)
Ct1 = np.bmat(([Ct1_00, Ct1_01, Ct1_02],[Ct1_10, Ct1_11, Ct1_12],[Ct1_20, Ct1_21, Ct1_22]))
#Ct1_stat = (Ct1 + Ct1.T) /2
L = (Ct1 - Ct1.T) / (2 * dt)
L_sym = (L - L.T) / 2 #Increase the statistic becasuse L is symmetrical
#More statistic using the time reversal propierty
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
L_stat = (L - epsilon.dot(L_sym).dot(epsilon)) / 2
#Change the format of L in order to obtain the derivative of C in each tiem step
Ctdev0 = L_stat
Ctdev[0,:] = np.bmat((Ctdev0[0:nodes, 0:nodes].reshape(1,nodes**2), Ctdev0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Ctdev0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , Ctdev0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Ctdev0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctdev0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Ctdev0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Ctdev0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Ctdev0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#######################################
#COMPUTE Ctdev*Ctnorm^-1 AT t=0
#######################################
CtdevCtnorminv0 = Ctdev0.dot(Ctnorminv0)
CtdevCtnorminv[0,:] = np.bmat((CtdevCtnorminv0[0:nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminv0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminv0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , CtdevCtnorminv0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminv0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminv0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , CtdevCtnorminv0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminv0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminv0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#######################################
#COMPUTE Ctdev*Ctnorm^-1-L AT t=0
#######################################
CtdevCtnorminvMinusL0 = CtdevCtnorminv0 - L_stat
CtdevCtnorminvMinusL[0,:] = np.bmat((CtdevCtnorminvMinusL0[0:nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , CtdevCtnorminvMinusL0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , CtdevCtnorminvMinusL0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtnorminvMinusL0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#####################################
#COMPUTE CtdevCt  AT t=0
#####################################
CtdevCtinv0 = L_stat.dot(R)
CtdevCtinv[0,:] = np.bmat((CtdevCtinv0[0:nodes, 0:nodes].reshape(1,nodes**2), CtdevCtinv0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtinv0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , CtdevCtinv0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtinv0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtinv0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , CtdevCtinv0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), CtdevCtinv0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), CtdevCtinv0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#####################################
#COMPUTE THE SECOND DERIVATIVE OF C AT t=0
#####################################
#Ct2dev0 = (Ct1 + Ct1.T - 2*Ct0_stat) / dt **2
##Change format: Matrix -> Vector
#Ct2dev[0,:] = np.bmat((Ct2dev0[0:nodes, 0:nodes].reshape(1,nodes**2), Ct2dev0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Ct2dev0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
#                , Ct2dev0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Ct2dev0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Ct2dev0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
#                    , Ct2dev0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Ct2dev0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Ct2dev0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#####################################
#CALCULATE D at t=0 as -2*L
#####################################
D0 = - 2 * L
#Change format: Matrix -> Vector
D[0,:] = np.bmat((D0[0:nodes, 0:nodes].reshape(1,nodes**2), D0[0:nodes, nodes:2*nodes].reshape(1,nodes**2), D0[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                , D0[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), D0[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), D0[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                    , D0[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), D0[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), D0[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#####################################################################################################################################
############################################## COMPUTE FOR t >0 ###################################################################
#####################################################################################################################################

for t in range(1, steps-1, 1):
#############
#D(t)
#############
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
    #Calculate the inverse of C
    Cinv = linalg.pinv(C, rcond = tol)
    #Select the blocks of the matrix C(t+dt)
    Cforward_00 = np.asmatrix(Ct[t+1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_01 = np.asmatrix(Ct[t+1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_02 = np.asmatrix(Ct[t+1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_10 = np.asmatrix(Ct[t+1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_11 = np.asmatrix(Ct[t+1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_12 = np.asmatrix(Ct[t+1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_20 = np.asmatrix(Ct[t+1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_21 = np.asmatrix(Ct[t+1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cforward_22 = np.asmatrix(Ct[t+1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
    #Create the matrix C at time t+dt
    Cforward = np.bmat(([Cforward_00, Cforward_01, Cforward_02],[Cforward_10, Cforward_11, Cforward_12],[Cforward_20, Cforward_21, Cforward_22]))
    #Select the blocks of the matrix C(t-dt)
    Cbackward_00 = np.asmatrix(Ct[t-1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_01 = np.asmatrix(Ct[t-1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_02 = np.asmatrix(Ct[t-1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_10 = np.asmatrix(Ct[t-1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_11 = np.asmatrix(Ct[t-1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_12 = np.asmatrix(Ct[t-1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_20 = np.asmatrix(Ct[t-1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_21 = np.asmatrix(Ct[t-1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
    Cbackward_22 = np.asmatrix(Ct[t-1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
    #Create the matrix C at time t-dt
    Cbackward = np.bmat(([Cbackward_00, Cbackward_01, Cbackward_02],[Cbackward_10, Cbackward_11, Cbackward_12],[Cbackward_20, Cbackward_21, Cbackward_22]))
    #Derive C at time t
    Cdev = (Cforward - Cbackward) / (2 * dt)
    #Compute D matrix as -(Cdev(t) + L*R*C(t)) * Cinv(t)*Rinv
    M = -np.asmatrix((Cdev + L.dot(R).dot(C)).dot(Cinv).dot(Ct0_stat))
    #Change format: Matrix -> Vector
    D[t,:] = np.bmat((M[0:nodes, 0:nodes].reshape(1,nodes**2), M[0:nodes, nodes:2*nodes].reshape(1,nodes**2), M[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
            , M[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), M[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), M[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                , M[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), M[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), M[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
#    #Calculate the 2nd derivative of C
#    C2dev =(Cforward + Cbackward - 2 * C) / dt**2
#    #Save the matrix C2dev at each time step
#    Ct2dev[t,:] = np.bmat((C2dev[0:nodes, 0:nodes].reshape(1,nodes**2), C2dev[0:nodes, nodes:2*nodes].reshape(1,nodes**2), C2dev[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
#            , C2dev[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), C2dev[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), C2dev[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
#                , C2dev[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), C2dev[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), C2dev[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

################
#C(t) normalized and its inverse
################
    #Calculate C normalized and change format: Matrix -> Vector
    Cnorm = R.dot(C)
    Cnorminv = linalg.pinv(Cnorm, rcond = tol)
    Ctnorm[t,:] = np.bmat((Cnorm[0:nodes, 0:nodes].reshape(1,nodes**2), Cnorm[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorm[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Cnorm[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Cnorm[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorm[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                        , Cnorm[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Cnorm[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorm[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

    Ctnorminv[t,:] = np.bmat((Cnorminv[0:nodes, 0:nodes].reshape(1,nodes**2), Cnorminv[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorminv[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                    , Cnorminv[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Cnorminv[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorminv[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                        , Cnorminv[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Cnorminv[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Cnorminv[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

################
#Cdev(t)*Cnorm(t) - L
################
    #Calculate and change format: Matrix -> Vector
    CdevCnorminvMinusL = Cdev.dot(Cnorminv) - L_stat
    CtdevCtnorminvMinusL[t,:] = np.bmat((CdevCnorminvMinusL[0:nodes, 0:nodes].reshape(1,nodes**2), CdevCnorminvMinusL[0:nodes, nodes:2*nodes].reshape(1,nodes**2), CdevCnorminvMinusL[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
                    , CdevCnorminvMinusL[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), CdevCnorminvMinusL[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), CdevCnorminvMinusL[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                        , CdevCnorminvMinusL[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), CdevCnorminvMinusL[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), CdevCnorminvMinusL[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

################
#Cdev(t)*Cinv(t)
################
    #Compute Cinv
    Cnorminv = linalg.pinv(Cnorm, rcond =tol)
    #Compute the product Cdev(t)*Cinv(t)
    P = Cdev.dot(Cinv)
    #Save the product at each time step
    CtdevCtnorminv[t,:] = np.bmat((P[0:nodes, 0:nodes].reshape(1,nodes**2), P[0:nodes, nodes:2*nodes].reshape(1,nodes**2), P[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
            , P[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), P[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), P[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                , P[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), P[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), P[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
    #Save the matrix Cinv at each time step
    Ctinv[t,:] = np.bmat((Cinv[0:nodes, 0:nodes].reshape(1,nodes**2), Cinv[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Cinv[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
            , Cinv[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Cinv[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Cinv[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                , Cinv[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Cinv[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Cinv[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
    #Save the matrix Cdev at each time step
    Ctdev[t,:] = np.bmat((Cdev[0:nodes, 0:nodes].reshape(1,nodes**2), Cdev[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Cdev[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
            , Cdev[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Cdev[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Cdev[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
                , Cdev[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Cdev[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Cdev[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))

#####################################################################################################################################
############################################## COMPUTES FOR LAST TIME STEP ###########################################################
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
#Cstep_00 = np.asmatrix(Ct[steps-1, 0:len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_01 = np.asmatrix(Ct[steps-1, len(Ct[0])/blocks:2 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_02 = np.asmatrix(Ct[steps-1, 2*len(Ct[0])/blocks:3 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_10 = np.asmatrix(Ct[steps-1, 3*len(Ct[0])/blocks:4 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_11 = np.asmatrix(Ct[steps-1, 4*len(Ct[0])/blocks:5 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_12 = np.asmatrix(Ct[steps-1, 5*len(Ct[0])/blocks:6 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_20 = np.asmatrix(Ct[steps-1, 6*len(Ct[0])/blocks:7 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_21 = np.asmatrix(Ct[steps-1, 7*len(Ct[0])/blocks:8 * len(Ct[0])/blocks].reshape(nodes, nodes))
#Cstep_22 = np.asmatrix(Ct[steps-1, 8*len(Ct[0])/blocks:9 * len(Ct[0])/blocks].reshape(nodes, nodes))
##Create the matrix C at last time step
#Cstep = np.bmat(([Cstep_00, Cstep_01, Cstep_02],[Cstep_10, Cstep_11, Cstep_12],[Cstep_20, Cstep_21, Cstep_22]))
##Calculate the inverse of C at last time step
#Cstepinv = linalg.pinv(Cstep, rcond = 1e-12)
##Change the format: Matrix -> Vector
#Cstep = np.bmat((Cstep[0:nodes, 0:nodes].reshape(1,nodes**2), Cstep[0:nodes, nodes:2*nodes].reshape(1,nodes**2), Cstep[0:nodes, 2*nodes:3*nodes].reshape(1,nodes**2)\
#          , Cstep[nodes:2*nodes, 0:nodes].reshape(1,nodes**2), Cstep[nodes:2*nodes, nodes:2*nodes].reshape(1,nodes**2), Cstep[nodes:2*nodes,2*nodes:3*nodes].reshape(1,nodes**2)\
#          , Cstep[2*nodes:3*nodes, 0:nodes].reshape(1,nodes**2), Cstep[2*nodes:3*nodes, nodes:2*nodes].reshape(1,nodes**2), Cstep[2*nodes:3*nodes, 2*nodes:3*nodes].reshape(1,nodes**2)))
##Copy Cstep to the last row of Ctinv
#Ctinv[steps-1,:] = Cstep


#####################################################################################################################################
############################################ COMPUTE THE INTEGRAL OF D ##############################################################
#####################################################################################################################################
M7_integral = np.sum(D[0:7,:], axis =0) * V * dt
D7_integral = np.bmat(([M7_integral[0:nodes**2].reshape(nodes,nodes), M7_integral[nodes**2:2*nodes**2].reshape(nodes,nodes), M7_integral[2*nodes**2:3*nodes**2].reshape(nodes,nodes)],[M7_integral[3*nodes**2:4*nodes**2].reshape(nodes,nodes), M7_integral[4*nodes**2:5*nodes**2].reshape(nodes,nodes), M7_integral[5*nodes**2:6*nodes**2].reshape(nodes,nodes)], [M7_integral[6*nodes**2:7*nodes**2].reshape(nodes,nodes), M7_integral[7*nodes**2:8*nodes**2].reshape(nodes,nodes), M7_integral[8*nodes**2:9*nodes**2].reshape(nodes,nodes)]))

#M7 = D[7,:]
#D7 = np.bmat(([M7[0:10000].reshape(nodes,nodes), M7[10000:20000].reshape(nodes,nodes), M7[20000:30000].reshape(nodes,nodes)],[M7[30000:40000].reshape(nodes,nodes), M7[40000:50000].reshape(nodes,nodes), M7[50000:60000].reshape(nodes,nodes)], [M7[60000:70000].reshape(nodes,nodes), M7[70000:80000].reshape(nodes,nodes), M7[80000:90000].reshape(nodes,nodes)]))

#####################################################################################################################################
###################################################### SAVE THE OUTPUT ##############################################################
#####################################################################################################################################
np.savetxt('D_rhoeg', D)
np.savetxt('D7_integral_rhoeg', D7_integral)
np.savetxt('Ct0_rhoeg', Ct0_stat)
np.savetxt('L_rhoeg',L_stat)
np.savetxt('R_rhoeg', R)
np.savetxt('Ctnorminv', Ctnorminv)
np.savetxt('Ctnorm', Ctnorm)
np.savetxt('Ctinv', Ctinv)
np.savetxt('Ctdev', Ctdev)
#np.savetxt('Ct2dev', Ct2dev)
np.savetxt('CtdevCtnorminv', CtdevCtnorminv)
np.savetxt('CtdevCtnorminvMinusL', CtdevCtnorminvMinusL)
#EOF
