import numpy as np
nNodes = 56
def pbc(C):
    Cstat = np.zeros((nNodes,nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            for k in range(nNodes):
                Cstat[i,j] += C[(i+k)%nNodes,(j+k)%nNodes]
    return Cstat/nNodes
cov = np.loadtxt('gxgx0')

cov_momentum=pbc(cov)

np.savetxt('gxgx0_simetrizado', cov_momentum)
