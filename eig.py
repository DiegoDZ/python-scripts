import numpy as np
from scipy import linalg

def step(x):
    return x * (x > 0)

Cl0       = np.loadtxt('Cl0')
C0        = np.loadtxt('C0')
Cl0inv    = linalg.pinv(Cl0, rcond = 1e-3)
Lambda    = np.dot(C0,Cl0inv)
LambdaSim = (Lambda + Lambda.T ) / 2
LambdaAnt = (Lambda - Lambda.T ) / 2

#Compute eigenvalues and eigenvectors
w,v       = linalg.eigh(LambdaSim)

#Reconstruc matrices
LambdaSimRec = 0
for i in range(len(w)):
   L = np.outer(v[:,i], v[:,i].T) * step(w[i])
   LambdaSimRec += L

#Errors
LambdaSimError = LambdaSim - LambdaSimRec
LambdaRec = LambdaSimRec + LambdaAnt

#Save output
np.savetxt('LambdaSim', LambdaSim)
np.savetxt('LambdaSimRec', LambdaSimRec)
np.savetxt('LambdaSimError', LambdaSimError)
np.savetxt('LambdaRec', LambdaRec)
np.savetxt('wLambda', w)
np.savetxt('vLambda', v)



#Cl0sim    = (Cl0 + Cl0.T ) / 2
#Cl0invsim = linalg.pinv(Cl0sim, rcond = 1e-3)
#Cl0invsim = (Cl0inv + Cl0inv.T ) / 2

#Compute eigenvalues and eigenvectors
#w,v       = linalg.eigh(Cl0sim)
#winv,vinv = linalg.eigh(Cl0invsim)

#Reconstruc matrices
#Cl0simrec = 0
#for i in range(len(w)):
#   C = np.outer(v[:,i], v[:,i].T) * step(w[i])
#   Cl0simrec += C
#Cl0simrecinv = linalg.pinv(Cl0simrec, rcond = 1e-3)
#
##Errors
#Cl0error  = Cl0sim - Cl0simrec
#Cl0inverror  = Cl0invsim - Cl0simrecinv

#Save output
#np.savetxt('wCl0', w)
#np.savetxt('vCl0', v)
#np.savetxt('wCl0inv', winv)
#np.savetxt('vCl0inv', vinv)
#np.savetxt('Cl0sim', Cl0sim)
#np.savetxt('Cl0invsim', Cl0invsim)
#np.savetxt('Cl0simrec', Cl0simrec)
#np.savetxt('Cl0simrecinv', Cl0simrecinv)
#np.savetxt('Cl0error', Cl0error)
#np.savetxt('Cl0inverror', Cl0inverror)


