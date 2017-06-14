#useful python functions

import numpy as np

#nBlocks = number correlations files
#sBlocks = int(np.sqrt(nBlocks))
#nNodes = number nodes
#dim = sBlocks * nNodes
#mu, nu = nodes selected


#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#Select the elements of a block matrix which correspond to the nodes mu,nu.
def selectNodes(A, mu, nu):
    B = np.array((A[mu,nu], A[mu,nu+nNodes], A[mu,nu+2*nNodes], A[mu+nNodes,nu], A[mu+nNodes,nu+nNodes], A[mu+nNodes,nu+2*nNodes], A[mu+2*nNodes,nu], A[mu+2*nNodes,nu+nNodes], A[mu+2*nNodes,nu+2*nNodes]))
    return B

