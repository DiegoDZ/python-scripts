###########################
###########################
# This script contains two subrutines: reshape_vm converts an input vector 
# into a matrix, and reshape_mv which does the oposite. 
#This is very useful when one has a time dependent matrix. Converting the matrix into a vector in each time step we can deal with a 2D array instead of 3D array. 
#
# Author: DiegoDZ
# Date:  June 2017
###########################

import numpy as np

nBlocks = XXXX
sBlocks = int(np.sqrt(nBlocks))
nNodes  = XXXX
dim     = sBlocks * nNodes
mu, nu  = XXXX 


#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

