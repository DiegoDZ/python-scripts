#------------------------------------------------------------------------------
#                            checkplateau.py
#------------------------------------------------------------------------------
# This script computes the matrix lambda
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : February 2018
#                         Modified : February 2018
#                         Run      : python checkplatau.py  (Run with python 2.7)
#------------------------------------------------------------------------------
import numpy as np
from scipy import linalg

#------------------------------------------------------------------------------
# Define global variables
#------------------------------------------------------------------------------
Lx,Ly,Lz    = 40.0,40.0,33.0  #dimensions of the simulation box   
totalNodes  = 66              #number of nodes
nNodes      = 61              #number of fluid nodes
dz          = Lz/totalNodes   #bin size
V           = dz * Lx * Ly    #bin volume
Temp        = 2.0             #temperature
dt          = 0.004           #lammps dt=2e-3 (but info saved every 2 steps)
#nSteps      = 7500            #t=30 (r.u.). The 'support' of the correlation files after cut them and the support of the C(t) predicted
nSteps      = 100
tol         = 1e-3            #rcond in linalg.pinv. It will be use to compute R
nBlocks     = 1
nVar        = int(np.sqrt(nBlocks))
dim         = nVar * nNodes

eps                                 = np.identity(dim)
eps[dim-nNodes:dim,dim-nNodes:dim] *= -1

F    = (1 / dz) * ((np.eye(nNodes)) - np.eye(nNodes, nNodes, 1))
Finv = linalg.pinv(F, rcond = tol)
B    = F.T
Binv = linalg.pinv(B, rcond = tol)

#------------------------------------------------------------------------------
# Subrutines
#------------------------------------------------------------------------------
#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(nVar,nVar,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(nVar,nNodes,nVar,nNodes).swapaxes(1,2).ravel()
    return B

#Calculate the derivative
def derivative(Ct,row):
    Cforward  = reshape_vm(Ct[row+1,:])
    Cbackward = reshape_vm(Ct[row-1,:])
    Cdev      = (Cforward - Cbackward) / (2 * dt)
    return Cdev

#------------------------------------------------------------------------------
# Load files
#------------------------------------------------------------------------------
Ct        = np.loadtxt('Ct-sf-200steps.dat')
c_SxzSxz  = np.loadtxt('corr-SxzSxz-sf-200steps.dat')
c_SxzFx   = np.loadtxt('corr-SxzFx-sf-200steps.dat')
c_FxSxz   = np.loadtxt('corr-FxSxz-sf-200steps.dat')
c_FxFx    = np.loadtxt('corr-FxFx-sf-200steps.dat')


""" Check efecto Finv
C0 = reshape_vm(Ct[0,:])
R  = linalg.pinv(C0, rcond = tol)

C_t02 = reshape_vm(Ct[50,:])
Cinv_t02 = linalg.pinv(C_t02, rcond = tol)
fatIdC = F.dot(Cinv_t02).dot(C0).dot(Finv)

np.savetxt('Cinvt02C0', Cinv_t02.dot(C0))
np.savetxt('Cinvt02C0Finv', Cinv_t02.dot(C0).dot(Finv))
np.savetxt('fatIdC', fatIdC)
np.savetxt('C0Finv', C0.dot(Finv))
"""

C0    = reshape_vm(Ct[0,:])
R     = linalg.pinv(C0, rcond = tol)
Cdev0 = (reshape_vm(Ct[1,:]) - reshape_vm(Ct[1,:]).T)/ (2 * dt)

#------------------------------------------------------------------------------
# Start computation
#------------------------------------------------------------------------------
CtInvNorm  = np.zeros((nSteps, nNodes**2))
Mt         = np.zeros((nSteps+1, nNodes**2))
MCinvt         = np.zeros((nSteps+1, nNodes**2))
MtR         = np.zeros((nSteps+1, nNodes**2))

etat      = np.zeros((nSteps, nNodes**2))
Gt        = np.zeros((nSteps, nNodes**2))
Ht        = np.zeros((nSteps, nNodes**2))
gammat    = np.zeros((nSteps, nNodes**2))
CtdevCheck= np.zeros((nSteps, nNodes**2))
#checkCtdev= np.zeros((nSteps, nNodes**2))    #tirar

etaStart    = np.zeros((nSteps, nNodes**2))
GStart      = np.zeros((nSteps, nNodes**2))
HStart      = np.zeros((nSteps, nNodes**2))
gammaStart  = np.zeros((nSteps, nNodes**2))
lambdaStart = np.zeros((nSteps, nNodes**2))

lambdaTaut = np.zeros((nSteps, nNodes**2))

diag_etaStar   = np.zeros((nNodes, nSteps))
diag_GStar     = np.zeros((nNodes, nSteps))
diag_HStar     = np.zeros((nNodes, nSteps))
diag_gammaStar = np.zeros((nNodes, nSteps))

C0      = reshape_vm(Ct[0,:])
Mt[0,:] = reshape_mv(-Cdev0)


# Meter dentro del siguiente bucle
Ctdev = np.zeros((nSteps, nNodes**2))
for j in range(nSteps-1):
    Cdev            = derivative(Ct,j+1)
    Ctdev[j+1,:]    = reshape_mv(Cdev)
Ctdev[0,:] = reshape_mv(Cdev0)

Mt  = -Ctdev
####################################

for i in range(nSteps):
    print i
    C               = reshape_vm(Ct[i,:])
    Cinv            = linalg.pinv(C, rcond = tol)
    CtInvNorm[i,:]  = reshape_mv(Cinv.dot(C0))
    #Cdev            = derivative(Ct,i+1)
    #M               = -Cdev
    #Mt[i+1,:]       = reshape_mv(M)
    MCinv          = reshape_vm(Mt[i,:]).dot(Cinv)
    MCinvt[i,:]    = reshape_mv(MCinv)
    MR             = reshape_vm(Mt[i,:]).dot(R)
    MtR[i,:]       = reshape_mv(MR)

    eta             = (1/Temp) * reshape_vm((np.trapz(c_SxzSxz[0:i,:], dx=dt, axis=0)))
    G               = (1/Temp) * reshape_vm((np.trapz(c_FxSxz[0:i,:] , dx=dt, axis=0)))
    H               = (1/Temp) * reshape_vm((np.trapz(c_SxzFx[0:i,:] , dx=dt, axis=0)))
    gamma           = (1/Temp) * reshape_vm((np.trapz(c_FxFx[0:i,:]  , dx=dt, axis=0)))
    etat[i,:]       = reshape_mv(eta)
    Ht[i,:]         = reshape_mv(H)
    Gt[i,:]         = reshape_mv(G)
    gammat[i,:]     = reshape_mv(gamma)

    CtdevCheck[i,:] = reshape_mv(-Temp * (B.dot(eta).dot(B.T) - G.dot(B.T) - B.dot(H) + gamma))

    #checkCtdev[i,:]  = Ctdev[i,:] - CtdevCheck[i,:]        #tirar

    etaStar         = eta.dot(F).dot(Cinv).dot(C0).dot(Finv)
    GStar           = G.dot(F).dot(Cinv).dot(C0).dot(Finv)
    HStar           = H.dot(Cinv).dot(C0)
    gammaStar       = gamma.dot(Cinv).dot(C0)
    etaStart[i,:]   = reshape_mv(etaStar)
    GStart[i,:]     = reshape_mv(GStar)
    HStart[i,:]     = reshape_mv(HStar)
    gammaStart[i,:] = reshape_mv(gammaStar)
    
    diag_etaStar[:,i]   = np.diag(etaStar)
    diag_GStar[:,i]     = np.diag(GStar)
    diag_HStar[:,i]     = np.diag(HStar)
    diag_gammaStar[:,i] = np.diag(gammaStar)

    lambdaStar       = Temp * (((F.T).dot(etaStar).dot(F) - GStar.dot(F) - (F.T).dot(HStar) + gammaStar).dot(R))
    lambdaStart[i,:] = reshape_mv(lambdaStar)



#------------------------------------------------------------------------------
# Save results
#------------------------------------------------------------------------------
np.savetxt('eta-t0.2.dat'  , reshape_vm(etat[50,:]))
np.savetxt('G-t0.2.dat'    , reshape_vm(Gt[50,:]))
np.savetxt('H-t0.2.dat'    , reshape_vm(Ht[50,:]))
np.savetxt('gamma-t0.2.dat', reshape_vm(gammat[50,:]))

np.savetxt('etaStar-t0.2.dat'  , reshape_vm(etaStart[50,:]))
np.savetxt('GStar-t0.2.dat'    , reshape_vm(GStart[50,:]))
np.savetxt('HStar-t0.2.dat'    , reshape_vm(HStart[50,:]))
np.savetxt('gammaStar-t0.2.dat', reshape_vm(gammaStart[50,:]))

np.savetxt('etaStarDiag.dat'  , diag_etaStar)
np.savetxt('GStarDiag.dat'    , diag_GStar)
np.savetxt('HStarDiag.dat'    , diag_HStar)
np.savetxt('gammaStarDiag.dat', diag_gammaStar)

np.savetxt('CinvNorm-t0.2.dat'  , reshape_vm(CtInvNorm[50,:]))
np.savetxt('M-t0.2.dat'      , reshape_vm(Mt[50,:]))
np.savetxt('CdevCheck-t0.2.dat' , reshape_vm(CtdevCheck[50,:]))
#np.savetxt('checkCtdev.dat'     , checkCtdev)   #tirar
np.savetxt('LambdaStar-t0.2.dat', reshape_vm(lambdaStart[50,:]))

np.savetxt('MCinv-t0.2.dat', reshape_vm(MCinvt[50,:]))
np.savetxt('MR-t0.02.dat', reshape_vm(MtR[5,:]))
np.savetxt('MR-t0.04.dat', reshape_vm(MtR[10,:]))
np.savetxt('MR-t0.06.dat', reshape_vm(MtR[15,:]))
np.savetxt('MR-t0.08.dat', reshape_vm(MtR[20,:]))
np.savetxt('MR-t0.1.dat',  reshape_vm(MtR[25,:]))
np.savetxt('MR-t0.2.dat',  reshape_vm(MtR[50,:]))
np.savetxt('MR-t0.3.dat',  reshape_vm(MtR[75,:]))

#EOF
