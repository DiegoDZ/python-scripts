#------------------------------------------------------------------------------
#                            buildCt.py
#------------------------------------------------------------------------------
# This script builds the matrix of correlations C(t) only with a bunch of
# correlations computing with LAMMPS.
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : January 2018
#                         Modified : January 2018
#                         Run      : python buildCt.py  (Run with python 2.7)
#------------------------------------------------------------------------------

import numpy as np
#----------------------------------
#Load files
#----------------------------------
block1 = np.loadtxt('corr-gxgx-sf-block1-AVG.dat')       #With this two blocks we will build the matrix of correlations C(t).
block2 = np.loadtxt('corr-gxgx-sf-block2-AVG.dat')

#----------------------------------
#Inputs
#----------------------------------
nSteps, nCols = np.shape(block1)
nNodes        = 61
nBlocks       = 1                                         #Do not confuse with the number of blocks we will use to build the matrix
                                                          #of correlations C(t). This number of blocks refers to the number of different
                                                          #correlation files (i.e. <gxgx(t)>, <gxrho(t)> ...).
                                                          #In this case the number of blocks is only one because the matrix of
                                                          #correlations will be <gxgx(t)>.
sBlocks       = int(np.sqrt(nBlocks))
dim           = sBlocks * nNodes
#We define the number of nodes for the blocks calculated with LAMMPS. Note the different between the number of nodes of C(t) and the number of nodes of block1 and block2.
nNodesBlock   = np.sqrt(nCols)
dimBlock      = sBlocks * nNodesBlock

#----------------------------------
#Subrutines
#----------------------------------
#We will distinguish between the reshapes for the block1 and block2 and the reshapes for the matrix of correlations C(t).

#Change format: vector-> matrix  (For block1 and block2)
def reshapeBlock_vm(A):
    B = A.reshape(nBlocks,nNodesBlock*nNodesBlock).reshape(sBlocks,sBlocks,nNodesBlock,nNodesBlock).transpose(0,2,1,3).reshape(dimBlock,dimBlock)
    return B

#Change format: matrix-> vector  (For blocks)
def reshapeBlock_mv(A):
    B = A.reshape(sBlocks,nNodesBlock,sBlocks,nNodesBlock).swapaxes(1,2).ravel()
    return B

#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#Build the matrix of correlations C. We will call this function every timestep.
def buildC(C,B):
    A = B[nNodesBlock/2, nNodesBlock/2:nNodesBlock]
    for i in range(int(nNodes-nNodesBlock+1)):
        C[nNodesBlock/2+i, nNodesBlock/2+i:nNodesBlock+i] = A
    return C

#Increase the statistics taking advantage of symmetries
def stat(A):
    B = (A.T + A) / 2
    C = (np.rot90(B,2).T + B) / 2
    return C

#----------------------------------
#Computation
#----------------------------------

Ct = np.zeros((nSteps,nNodes**2))

for t in range(nSteps):
    print "Building C("+str(t)+")"
    Z = np.zeros((nNodes, nNodes))
    B1 = reshapeBlock_vm(block1[t,:])
    B2 = reshapeBlock_vm(block2[t,:])
    Z[0:nNodesBlock,0:nNodesBlock]                          = B1         #Paste the blocks calculated with LAMMPS in the zero matrix Z.
    Z[nNodes-nNodesBlock:nNodes, nNodes-nNodesBlock:nNodes] = B2
    Cupper = buildC(Z,B1)                                                #Build the matrix C above the main diagonal.
    Crot = np.rot90(Cupper,2)                                            #Rotate tha matrix C to use the same subrutine to buil the
                                                                         #matrix C below the main diagonal.
    C = buildC(Crot,B2)
    Cstat = stat(C)                                                      #Increase the statistics.
    Ct[t,:] = reshape_mv(Cstat)

np.savetxt('Ct', Ct)
np.savetxt('C-t0',   reshape_vm(Ct[0,:]))
np.savetxt('C-t0.2', reshape_vm(Ct[49,:]))
np.savetxt('C-t0.4', reshape_vm(Ct[99,:]))

#EOF
