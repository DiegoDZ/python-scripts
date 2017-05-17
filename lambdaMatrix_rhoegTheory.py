##############################################
#
#
# AUTHOR: @DiegoDZ
# DATE: APRIL 2017
#
# run: >>> python lambdaMatrix_rhoegTheory.py
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
Ut = np.zeros((steps-1, nodes ** 2 * blocks))
wt = np.zeros((steps-1, 3 * nodes))
vt = np.zeros(( 3 * nodes * (steps-1), 3 * nodes))
wCt = np.zeros((steps-1, 3 * nodes))
vCt = np.zeros(( 3 * nodes * (steps-1), 3 * nodes))
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
#COMPUTE C(t=0)
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
Ct0_stat = (Ct0 + Ct0.T) / 2 #Increase the statistic because C(t=0) is symmetric
#########Compute R and Ctinv(t=0)########
R = linalg.pinv(Ct0_stat, rcond = tol)
#########Compute U(t=0) as C(t=0)*R########
Ut0 = Ct0_stat.dot(R)
Ut[0,:] = reshape_mv(Ut0)
#########Compute the eigenvalues and the eigenvectors of U(t=0)########
w0,v0 = linalg.eig(Ut0)
wt[0,:] = w0
vt[0:3*nodes,:] = v0
#########Compute the eigenvalues and the eigenvectors of C(t=0)########
wC0, vC0 = linalg.eig(Ct0_stat)
wCt[0,:] = wC0
vCt[0:3*nodes,:] = vC0


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

    #####################################
    # COMPUTE U(t)
    #####################################
        U = C.dot(R)
        Ut[t,:] = reshape_mv(U)
    #####################################
    # COMPUTE the eigenvalues and the eigenvectors of U(t)
    #####################################
        w,v = linalg.eig(U)
        wt[t,:] = w
        vt[t*3*nodes: (t*3*nodes) + 3*nodes,:] = v
    #####################################
    # COMPUTE the eigenvalues and the eigenvectors of C(t)
    #####################################
        wC,vC = linalg.eig(C)
        wCt[t,:] = wC
        vCt[t*3*nodes: (t*3*nodes) + 3*nodes,:] = vC

############################################ COMPUTE THE INTEGRAL OF M ##############################################################

#M7_integral = np.sum(Mt[0:7,:], axis =0) * V * dt
#M7_integral = np.bmat(([M7_integral[0:nodes**2].reshape(nodes,nodes), M7_integral[nodes**2:2*nodes**2].reshape(nodes,nodes), M7_integral[2*nodes**2:3*nodes**2].reshape(nodes,nodes)],[M7_integral[3*nodes**2:4*nodes**2].reshape(nodes,nodes), M7_integral[4*nodes**2:5*nodes**2].reshape(nodes,nodes), M7_integral[5*nodes**2:6*nodes**2].reshape(nodes,nodes)], [M7_integral[6*nodes**2:7*nodes**2].reshape(nodes,nodes), M7_integral[7*nodes**2:8*nodes**2].reshape(nodes,nodes), M7_integral[8*nodes**2:9*nodes**2].reshape(nodes,nodes)]))

###################################################### SAVE THE OUTPUT ##############################################################

#np.savetxt('R_rhoeg', R)
np.savetxt('Ut_rhoeg', Ut)
np.savetxt('eigenvaluesUt', wt)
np.savetxt('eigenvectorsUt', vt)
np.savetxt('eigenvaluesCt', wCt)
np.savetxt('eigenvectorsCt', vCt)

#EOF
