################################################
# This script takes advange of the simmetries of an input matrix of correlations in order to increase the statistics. 
# Author: DiegoDZ
# Date:   January 2017
################################################

import numpy as np
nNodes = 
def pbc(C):
    Cstat = np.zeros((nNodes,nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            for k in range(nNodes):
                Cstat[i,j] += C[(i+k)%nNodes,(j+k)%nNodes]
    return Cstat/nNodes
cov = np.loadtxt('gxgx0')

cov_momentum=pbc(cov)

np.savetxt('gxgx0', cov_momentum)
