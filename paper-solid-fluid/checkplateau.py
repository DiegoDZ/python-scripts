#------------------------------------------------------------------------------
#                            checkplateau.py
#------------------------------------------------------------------------------
# This script computes the matrix lambda
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : February 2018
#                         Modified : February 2018
#                         Run      : python checkplateau.py  (Run with python 2.7)
#------------------------------------------------------------------------------
import numpy as np
from scipy import linalg
from scipy.linalg import expm
from scipy.linalg import sqrtm
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
#nSteps      = 7499            #t=30 (r.u.). The 'support' of the correlation files after cut them and the support of the C(t) predicted
nSteps      = 500
tol         = 1e-3            #rcond in linalg.pinv. It will be use to compute R
nBlocks     = 1
nVar        = int(np.sqrt(nBlocks))
dim         = int(nVar * nNodes)

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
#Ct       = np.loadtxt('Ct-sf.dat')
#c_SxzSxz = np.loadtxt('corr-SxzSxz-sf.dat')
#c_SxzFx  = np.loadtxt('corr-SxzFx-sf.dat')
#c_FxSxz  = np.loadtxt('corr-FxSxz-sf.dat')
#c_FxFx   = np.loadtxt('corr-FxFx-sf.dat')
Ct       = np.loadtxt('Ct-sf-500steps.dat')
c_SxzSxz = np.loadtxt('corr-SxzSxz-sf-500steps.dat')
c_SxzFx  = np.loadtxt('corr-SxzFx-sf-500steps.dat')
c_FxSxz  = np.loadtxt('corr-FxSxz-sf-500steps.dat')
c_FxFx   = np.loadtxt('corr-FxFx-sf-500steps.dat')


C0      = reshape_vm(Ct[0,:])
R       = linalg.pinv(C0, rcond = tol)
Cdev0   = (reshape_vm(Ct[1,:]) - reshape_vm(Ct[1,:]).T)/ (2 * dt)

#------------------------------------------------------------------------------
# Start computation
#------------------------------------------------------------------------------
CtInv = np.zeros((nSteps, nNodes**2))
CtInvNorm = np.zeros((nSteps, nNodes**2))
CtInvNorm = np.zeros((nSteps, nNodes**2))
Ctnorm = np.zeros((nSteps, nNodes**2))
RCt       = np.zeros((nSteps, nNodes**2))
#FetaFt  = np.zeros((nSteps, nNodes**2))
#GFt  = np.zeros((nSteps, nNodes**2))
#FHt  = np.zeros((nSteps, nNodes**2))
checkDev  = np.zeros((nSteps, nNodes**2))
checkDevStar  = np.zeros((nSteps, nNodes**2))
Mt        = np.zeros((nSteps, nNodes**2))
gt        = np.zeros(nSteps)

etat      = np.zeros((nSteps, nNodes**2))
Gt        = np.zeros((nSteps, nNodes**2))
Ht        = np.zeros((nSteps, nNodes**2))
gammat    = np.zeros((nSteps, nNodes**2))
Lambdat = np.zeros((nSteps, nNodes**2))
etaRC      = np.zeros((nSteps, nNodes**2))
GRC        = np.zeros((nSteps, nNodes**2))
HRC        = np.zeros((nSteps, nNodes**2))
gammaRC    = np.zeros((nSteps, nNodes**2))
#etaModift      = np.zeros((nSteps, nNodes**2))
#GModift      = np.zeros((nSteps, nNodes**2))
#HModift      = np.zeros((nSteps, nNodes**2))
#gammaModift      = np.zeros((nSteps, nNodes**2))
#CtdevCheck= np.zeros((nSteps, nNodes**2))
#CdevCinvC0t= np.zeros((nSteps, nNodes**2))
#CdevCinvt= np.zeros((nSteps, nNodes**2))
#checkCtdev= np.zeros((nSteps, nNodes**2))    #tirar

etaStar    = np.zeros((nSteps, nNodes**2))
GStar      = np.zeros((nSteps, nNodes**2))
HStar      = np.zeros((nSteps, nNodes**2))
gammaStar  = np.zeros((nSteps, nNodes**2))
etaStarF   = np.zeros((nSteps, nNodes**2))
GStarF     = np.zeros((nSteps, nNodes**2))

LambdaStar = np.zeros((nSteps, nNodes**2))

#lambdaTaut = np.zeros((nSteps, nNodes**2))

#diag_etaStar    = np.zeros((nNodes, nSteps))
#diag_GStar      = np.zeros((nNodes, nSteps))
#diag_HStar      = np.zeros((nNodes, nSteps))
#diag_gammaStar  = np.zeros((nNodes, nSteps))
#diag_etaModif   = np.zeros((nNodes, nSteps))
#diag_GModif     = np.zeros((nNodes, nSteps))
#diag_HModif     = np.zeros((nNodes, nSteps))
#diag_gammaModif = np.zeros((nNodes, nSteps))

# Meter dentro del siguiente bucle
Ctdev     = np.zeros((nSteps, nNodes**2))
CtdevDiag = np.zeros((nNodes, nSteps))
for j in range(nSteps-1):
    Cdev            = derivative(Ct,j+1)
    Ctdev[j+1,:]    = reshape_mv(Cdev)
    CtdevDiag[:,j+1]  = np.diag(Cdev)
Ctdev[0,:] = reshape_mv(Cdev0)
CtdevDiag[:,0] = np.diag(Cdev0)

np.savetxt('CtDev-sf.dat'      , Ctdev)
np.savetxt('CtDevDiag-sf.dat'  , CtdevDiag)
np.savetxt('CDev-sf-t0.06.dat', reshape_vm(Ctdev[15,:]))

np.savetxt('CDev-sf-t0.1.dat', reshape_vm(Ctdev[25,:]))
np.savetxt('CDev-sf-t0.2.dat', reshape_vm(Ctdev[50,:]))
np.savetxt('CDev-sf-t0.4.dat', reshape_vm(Ctdev[100,:]))
np.savetxt('CDev-sf-t0.8.dat', reshape_vm(Ctdev[200,:]))
np.savetxt('CDev-sf-t1.6.dat', reshape_vm(Ctdev[400,:]))

####################################
for i in range(nSteps):
    print i
    C               = reshape_vm(Ct[i,:])
    Cinv            = linalg.pinv(C, rcond = tol)
    CtInv[i,:]      = reshape_mv(Cinv)
    #Cinv            = inv(C) 
    CinvNorm        = Cinv.dot(C0)
    sqrtCinvNorm    = sqrtm(CinvNorm)
    CinvNormT       = CinvNorm.T
    CtInvNorm[i,:]  = reshape_mv(CinvNorm)
    RC              = R.dot(C)
    RCt[i,:]        = reshape_mv(RC)
    #g               = (1/float(nNodes)) * np.trace(RC)
    #gt[i]           = g
    
    M          = (-1/Temp)*(reshape_vm(Ctdev[i,:]).dot(Cinv).dot(C0))
    Mt[i,:]    = reshape_mv(M)
    #MR             = reshape_vm(Mt[i,:]).dot(R)
    #MtR[i,:]       = reshape_mv(MR)
    #MStar[i,:]  = - reshape_mv(reshape_vm(Ctdev[i,:]).dot(CinvNorm))

    eta            = (1/Temp) * reshape_vm((np.trapz(c_SxzSxz[0:i,:], dx=dt, axis=0)))
    G              = (1/Temp) * reshape_vm((np.trapz(c_FxSxz[0:i,:] , dx=dt, axis=0)))
    H              = (1/Temp) * reshape_vm((np.trapz(c_SxzFx[0:i,:] , dx=dt, axis=0)))
    #GPrev          = (1/Temp) * reshape_vm((np.trapz(c_FxSxz[0:i,:] , dx=dt, axis=0)))
    #HPrev          = (1/Temp) * reshape_vm((np.trapz(c_SxzFx[0:i,:] , dx=dt, axis=0)))
    #G              = 0.5*(GPrev + HPrev.T)
    #H              = 0.5*(GPrev.T + HPrev)
    gamma         = (1/Temp) * reshape_vm((np.trapz(c_FxFx[0:i,:]  , dx=dt, axis=0)))
    #etaRC[i,:]    = reshape_mv(eta.dot(RC))
    #GRC[i,:]      = reshape_mv(G.dot(RC))
    #HRC[i,:]      = reshape_mv(H.dot(RC))
    #gammaRC[i,:]  = reshape_mv(gamma.dot(RC))

    etat[i,:]      = reshape_mv(eta)
    Gt[i,:]        = reshape_mv(G)
    Ht[i,:]        = reshape_mv(H)
    gammat[i,:]    = reshape_mv(gamma)
    Lambda         = Temp * (((F.T).dot(eta).dot(F) - G.dot(F) - (F.T).dot(H) + gamma).dot(R))
    Lambdat[i,:]   = reshape_mv(Lambda)
    #MStar2[i,:]    = reshape_mv((F.T).dot(eta).dot(F) - G.dot(F) - (F.T).dot(H) + gamma)
    etaStar[i,:]   = reshape_mv(eta.dot(F).dot(Cinv).dot(C0).dot(Finv))
    GStar[i,:]     = reshape_mv(G.dot(F).dot(Cinv).dot(C0).dot(Finv))
    HStar[i,:]     = reshape_mv(H.dot(Cinv).dot(C0))
    gammaStar[i,:] = reshape_mv(gamma.dot(Cinv).dot(C0))
    etaStarF[i,:]   = reshape_mv(eta.dot(F).dot(Cinv).dot(C0))
    GStarF[i,:]     = reshape_mv(G.dot(F).dot(Cinv).dot(C0))
    
    #FetaF       = (F.T).dot(eta).dot(F)
    #GF          = G.dot(F)
    #FH          = (F.T).dot(H)
    #FetaFt[i,:] = reshape_mv(FetaF)
    #GFt[i,:]    = reshape_mv(GF)
    #FHt[i,:]    = reshape_mv(FH)

    #check          = - Temp * ((F.T).dot(eta).dot(F) - G.dot(F) - (F.T).dot(H) + gamma)
    #checkDev[i,:]  = reshape_mv(check)

    #etaStar        = eta.dot(CinvNorm)
    #etaStar        = (sqrtCinvNorm.T).dot(eta).dot(sqrtCinvNorm)
    #etaStar        = (1/g)*eta
    #etaStart[i,:]  = reshape_mv(etaStar)
    #GStarPrev      = (1/g)*G
    #HStarPrev      = (1/g)*H
    #GStarPrev        = (sqrtCinvNorm.T).dot(G).dot(sqrtCinvNorm)
    #HStarPrev        = (sqrtCinvNorm.T).dot(H).dot(sqrtCinvNorm)
    #GStarPrev      = G.dot(CinvNorm)
    #HStarPrev      = H.dot(CinvNorm)
    #GStar          = 0.5*(GStarPrev + HStarPrev.T)
    #HStar          = 0.5*(GStarPrev.T + HStarPrev)
    #GStart[i,:]    = reshape_mv(GStar)
    #HStart[i,:]    = reshape_mv(HStar)
    #gammaStar       = (1/g)*gamma
    #gammaStar       = gamma.dot(CinvNorm)
    #gammaStar        = (sqrtCinvNorm.T).dot(gamma).dot(sqrtCinvNorm)
    #gammaStart[i,:] = reshape_mv(gammaStar)

    #LambdaStar       = Temp * (((F.T).dot(etaStar).dot(F) - GStar.dot(F) - (F.T).dot(HStar) + gammaStar).dot(R))
    #LambdaStart[i,:] = reshape_mv(LambdaStar)
    #factorEta[i,:]   = reshape_mv((F.T).dot(eta).dot(F))
    #factorH[i,:]     = reshape_mv((F.T).dot(H))
    #factorG[i,:]     = reshape_mv(G.dot(F))

    #checkDevStar[i,:]   = -Temp * reshape_mv(LambdaStar.dot(C))
    #
    #diag_etaStar[:,i]   = np.diag(etaStar)
    #diag_GStar[:,i]     = np.diag(GStar)
    #diag_HStar[:,i]     = np.diag(HStar)
    #diag_gammaStar[:,i] = np.diag(gammaStar)

    #lambdaStar         = Temp * (((F.T).dot(etaStar).dot(F) - GStar.dot(F) - (F.T).dot(HStar) + gammaStar).dot(R))
    #alfa               = (1/float(nNodes)) * np.sum(lambdaStar)
    #lambdaStart[i,:]   = reshape_mv(lambdaStar)

    #etaModif             = eta*np.exp(alfa*dt*i)
    #GModif               = G*np.exp(alfa*dt*i)
    #HModif               = H*np.exp(alfa*dt*i)
    #gammaModif           = gamma*np.exp(alfa*dt*i)
    #etaModift[i,:]       = reshape_mv(etaModif)
    #GModift[i,:]         = reshape_mv(GModif)
    #HModift[i,:]         = reshape_mv(HModif)
    #gammaModift[i,:]     = reshape_mv(gammaModif)
    #diag_etaModif[:,i]   = np.diag(etaModif)
    #diag_GModif[:,i]     = np.diag(GModif)
    #diag_HModif[:,i]     = np.diag(HModif)
    #diag_gammaModif[:,i] = np.diag(gammaModif)

#------------------------------------------------------------------------------
# Save results
#------------------------------------------------------------------------------

#np.savetxt('FetaF-t0.2.dat'  , reshape_vm(FetaFt[50,:]))
#np.savetxt('GF-t0.2.dat'    ,  reshape_vm(GFt[50,:]))
#np.savetxt('FH-t0.2.dat'    ,  reshape_vm(FHt[50,:]))
#np.savetxt('FetaF-t1.6.dat'  , reshape_vm(FetaFt[400,:]))
#np.savetxt('GF-t1.6.dat'    ,  reshape_vm(GFt[400,:]))
#np.savetxt('FH-t1.6.dat'    ,  reshape_vm(FHt[400,:]))
#np.savetxt('gamma-t1.6.dat', reshape_vm(gammat[400,:]))

np.savetxt('eta.dat'  , etat)
np.savetxt('G.dat'    , Gt)
np.savetxt('H.dat'    , Ht)
np.savetxt('gamma.dat', gammat)
np.savetxt('CtInv-sf.dat', CtInv)


np.savetxt('eta-t0.06.dat'  , reshape_vm(etat[15,:]))
np.savetxt('G-t0.06.dat'    , reshape_vm(Gt[15,:]))
np.savetxt('H-t0.06.dat'    , reshape_vm(Ht[15,:]))
np.savetxt('gamma-t0.06.dat', reshape_vm(gammat[15,:]))

np.savetxt('eta.mu30nu30.dat'  , etat[:,1798])
np.savetxt('G-mu3nu3.dat'    , Gt[:,124])
np.savetxt('H-mu3nu3.dat'    , Ht[:,124])
np.savetxt('gamma-mu3nu3.dat', gammat[:,124])
np.savetxt('eta-t0.2.dat'  , reshape_vm(etat[50,:]))
np.savetxt('G-t0.2.dat'    , reshape_vm(Gt[50,:]))
np.savetxt('H-t0.2.dat'    , reshape_vm(Ht[50,:]))
np.savetxt('gamma-t0.2.dat', reshape_vm(gammat[50,:]))

#np.savetxt('etaRC.dat'  , etaRC)
#np.savetxt('GRC.dat'    , GRC)
#np.savetxt('HRC.dat'    , HRC)
#np.savetxt('gammaRC.dat', gammaRC)
#np.savetxt('etaRC-mu30nu30.dat'  , etaRC[:,1798])
#np.savetxt('GRC-mu3nu3.dat'    , GRC[:,124])
#np.savetxt('HRC-mu3nu3.dat'    , HRC[:,124])
#np.savetxt('gammaRC-mu3nu3.dat', gammaRC[:,124])

np.savetxt('etaStar.dat'  , etaStar)
np.savetxt('GStar.dat'    , GStar)
np.savetxt('HStar.dat'    , HStar)
np.savetxt('gammaStar.dat', gammaStar)
np.savetxt('etaStar-mu30nu30.dat', etaStar[:,1798])
np.savetxt('etaStar-mu30nu32.dat', etaStar[:,1800])
np.savetxt('GStar-mu3nu3.dat'    , GStar[:,124])
np.savetxt('GStar-mu3nu2.dat'    , GStar[:,123])
np.savetxt('HStar-mu3nu3.dat'    , HStar[:,124])
np.savetxt('HStar-mu3nu2.dat'    , HStar[:,123])
np.savetxt('gammaStar-mu3nu2.dat', gammaStar[:,123])
np.savetxt('gammaStar-mu3nu3.dat', gammaStar[:,124])
np.savetxt('etaStar-t0.1.dat'  , reshape_vm(etaStar[25,:]))
np.savetxt('etaStar-t0.2.dat'  , reshape_vm(etaStar[50,:]))
np.savetxt('etaStar-t0.3.dat'  , reshape_vm(etaStar[75,:]))
np.savetxt('GStar-t0.1.dat'    , reshape_vm(GStar[25,:]))
np.savetxt('GStar-t0.2.dat'    , reshape_vm(GStar[50,:]))
np.savetxt('GStar-t0.3.dat'    , reshape_vm(GStar[75,:]))
np.savetxt('HStar-t0.1.dat'    , reshape_vm(HStar[25,:]))
np.savetxt('HStar-t0.2.dat'    , reshape_vm(HStar[50,:]))
np.savetxt('HStar-t0.3.dat'    , reshape_vm(HStar[75,:]))
np.savetxt('gammaStar-t0.04.dat', reshape_vm(gammaStar[10,:]))
np.savetxt('gammaStar-t0.1.dat', reshape_vm(gammaStar[25,:]))
np.savetxt('gammaStar-t0.14.dat', reshape_vm(gammaStar[35,:]))
np.savetxt('gammaStar-t0.2.dat', reshape_vm(gammaStar[50,:]))
np.savetxt('gammaStar-t0.3.dat', reshape_vm(gammaStar[75,:]))

np.savetxt('etaStarF-t0.2.dat'    , reshape_vm(etaStarF[50,:]))
np.savetxt('GStarF-t0.2.dat'      , reshape_vm(GStarF[50,:]))
np.savetxt('etaStarF-mu30nu30.dat', etaStarF[:,1798])
np.savetxt('etaStarF-mu30nu32.dat', etaStarF[:,1800])
np.savetxt('GStarF-mu3nu3.dat'    ,  GStarF[:,124])
np.savetxt('GStarF-mu3nu2.dat'    ,  GStarF[:,123])
#viernes
#np.savetxt('factorEta.dat'  , factorEta)
#np.savetxt('factorG.dat'    , factorG)
#np.savetxt('factorH.dat'    , factorH)
#np.savetxt('factorGamma.dat', factorGamma)
#np.savetxt('factorEta-mu30nu30.dat'  , factorEta[:,1798])
#np.savetxt('factorEta-mu3nu3.dat'  , factorEta[:,124])
#np.savetxt('factorG.dat'    , factorG)
#np.savetxt('factorG-mu3nu3.dat'  , factorG[:,124])
#np.savetxt('factorH.dat'    , factorH)
#np.savetxt('factorH-mu3nu3.dat'  , factorH[:,124])
#np.savetxt('factorGamma-mu3nu3.dat'  , factorGamma[:,124])
#viernes

#np.savetxt('factorEta-t0.04.dat'  , reshape_vm(factorEta[10,:]))
#np.savetxt('factorEta-t0.1.dat'   , reshape_vm(factorEta[25,:]))
#np.savetxt('factorEta-t0.2.dat'   , reshape_vm(factorEta[50,:]))
#np.savetxt('factorEta-t0.3.dat'   , reshape_vm(factorEta[75,:]))
#np.savetxt('factorG-t0.04.dat'    , reshape_vm(factorG[10,:]))
#np.savetxt('factorG-t0.1.dat'     , reshape_vm(factorG[25,:]))
#np.savetxt('factorG-t0.2.dat'     , reshape_vm(factorG[50,:]))
#np.savetxt('factorG-t0.3.dat'     , reshape_vm(factorG[75,:]))
#np.savetxt('factorH-t0.04.dat'    , reshape_vm(factorH[10,:]))
#np.savetxt('factorH-t0.1.dat'     , reshape_vm(factorH[25,:]))
#np.savetxt('factorH-t0.2.dat'     , reshape_vm(factorH[50,:]))
#np.savetxt('factorH-t0.3.dat'     , reshape_vm(factorH[75,:]))
#np.savetxt('factorGamma-t0.04.dat', reshape_vm(factorGamma[10,:]))
#np.savetxt('factorGamma-t0.1.dat' , reshape_vm(factorGamma[25,:]))
#np.savetxt('factorGamma-t0.2.dat' , reshape_vm(factorGamma[50,:]))
#np.savetxt('factorGamma-t0.3.dat' , reshape_vm(factorGamma[75,:]))

#np.savetxt('etaModif-t0.2.dat', reshape_vm(etaModift[50,:]))
#np.savetxt('GModif-t0.2.dat', reshape_vm(GModift[50,:]))
#np.savetxt('HModif-t0.2.dat', reshape_vm(HModift[50,:]))
#np.savetxt('gammaModif-t0.2.dat', reshape_vm(gammaModift[50,:]))

#np.savetxt('etaModif.dat'  , etaModift)
#np.savetxt('etagInv.dat'  , etagInvt)
#np.savetxt('GModif.dat'    , GModift)
#np.savetxt('HModif.dat'    , HModift)
#np.savetxt('gammaModif.dat', gammaModift)

#np.savetxt('etaStarDiag.dat'  , diag_etaStar)
#np.savetxt('GStarDiag.dat'    , diag_GStar)
#np.savetxt('HStarDiag.dat'    , diag_HStar)
#np.savetxt('gammaStarDiag.dat', diag_gammaStar)
#np.savetxt('etaModifDiag.dat' , diag_etaModif)
#np.savetxt('GModifDiag.dat' , diag_GModif)
#np.savetxt('HModifDiag.dat' , diag_HModif)
#np.savetxt('gammaModifDiag.dat' , diag_gammaModif)

np.savetxt('CtInvNorm-sf.dat'  , CtInvNorm)
#np.savetxt('CinvNorm-t0.2.dat'  , reshape_vm(CtInvNorm[50,:]))
#np.savetxt('MStar.dat' , MStar)
#np.savetxt('MStar2.dat', MStar2)
#np.savetxt('MStar-t0.04.dat' , reshape_vm(MStar[10,:]))
#np.savetxt('MStar2-t0.04.dat', reshape_vm(MStar2[10,:]))
#np.savetxt('MStar-t0.1.dat'  , reshape_vm(MStar[25,:]))
#np.savetxt('MStar2-t0.1.dat' , reshape_vm(MStar2[25,:]))
#np.savetxt('MStar-t0.14.dat' , reshape_vm(MStar[35,:]))
#np.savetxt('MStar2-t0.14.dat', reshape_vm(MStar2[35,:]))
#np.savetxt('MStar-t0.2.dat' , reshape_vm(MStar[50,:]))
#np.savetxt('MStar2-t0.2.dat', reshape_vm(MStar2[50,:]))
#np.savetxt('MStar-t0.4.dat' , reshape_vm(MStar[100,:]))
#np.savetxt('MStar2-t0.4.dat', reshape_vm(MStar2[100,:]))
np.savetxt('M.dat', Mt)
#np.savetxt('Lambda-mu30nu30.dat', Lambdat[:,1798])
#np.savetxt('Lambda-mu3nu3.dat'  , Lambdat[:,124])
#np.savetxt('LambdaStar-mu30nu30.dat', LambdaStart[:,1798])
#np.savetxt('LambdaStar-mu3nu3.dat'  , LambdaStart[:,124])
#np.savetxt('LambdaStar-t0.1.dat', reshape_vm(LambdaStart[25,:]))
#np.savetxt('LambdaStar-t0.2.dat', reshape_vm(LambdaStart[50,:]))
#np.savetxt('LambdaStar-t0.3.dat', reshape_vm(LambdaStart[75,:]))
np.savetxt('Lambda-t0.06.dat', reshape_vm(Lambdat[15,:]))

#np.savetxt('gt.dat', gt)
#np.savetxt('RCt.dat', RCt)
#np.savetxt('RC-t0.2.dat', reshape_vm(RCt[50,:]))
#
#np.savetxt('checkDevStar-t0.1.dat', reshape_vm(checkDevStar[25,:]))
#np.savetxt('checkDevStar-t0.2.dat', reshape_vm(checkDevStar[50,:]))
#np.savetxt('checkDevStar-t0.4.dat', reshape_vm(checkDevStar[100,:]))
#np.savetxt('checkDev-t0.1.dat', reshape_vm(checkDev[25,:]))
#np.savetxt('checkDev-t0.2.dat', reshape_vm(checkDev[50,:]))
#np.savetxt('checkDev-t0.4.dat', reshape_vm(checkDev[100,:]))
#np.savetxt('checkDev-t0.8.dat', reshape_vm(checkDev[200,:]))
#np.savetxt('checkDev-t1.6.dat', reshape_vm(checkDev[400,:]))

#np.savetxt('MCinv-t0.2.dat', reshape_vm(MCinvt[50,:]))
#np.savetxt('MR-t0.02.dat', reshape_vm(MtR[5,:]))
#np.savetxt('MR-t0.04.dat', reshape_vm(MtR[10,:]))
#np.savetxt('MR-t0.06.dat', reshape_vm(MtR[15,:]))
#np.savetxt('MR-t0.08.dat', reshape_vm(MtR[20,:]))
#np.savetxt('MR-t0.1.dat',  reshape_vm(MtR[25,:]))
#np.savetxt('MR-t0.2.dat',  reshape_vm(MtR[50,:]))
#np.savetxt('MR-t0.3.dat',  reshape_vm(MtR[75,:]))

#Predicciones
#tau = 0.084
tau = 0.06
#C(t) predicted

nSteps = 6000
def computeCtpredict(Ct,Lambda):
    Ctpredict      = np.zeros((nSteps, nNodes**2*nBlocks))
    row = int(round(tau / dt))
    t   = 0
    for j in range(nSteps):
        print tau, j
        Ctpredict[j,:]    = reshape_mv(np.dot(expm(-1.5*Lambda * (t-tau)), reshape_vm(Ct[row])))
        t+=dt
    return Ctpredict
#c_gxgx_predicted = computeCtpredict(Ct, reshape_vm(LambdaStart[21,:]))
c_gxgx_predicted = computeCtpredict(Ct, reshape_vm(Lambdat[15,:]))


#np.savetxt('corr-gxgx-predicted.dat', c_gxgx_predicted)
#np.savetxt('corr-gxgx-predicted-LambdaStarn-tau0.12-mu30nu30.dat', c_gxgx_predicted[:,1798])
#np.savetxt('corr-gxgx-predicted-LambdaStarn-tau0.084-mu3nu3.dat'  , c_gxgx_predicted[:,124])
np.savetxt('corr-gxgx-predicted-Lambdat0.06Factor1.5-mu30nu30.dat', c_gxgx_predicted[:,1798])
np.savetxt('corr-gxgx-predicted-Lambdat0.06Factor1.5-mu30nu33.dat', c_gxgx_predicted[:,1801])
np.savetxt('corr-gxgx-predicted-Lambdat0.06Factor1.5-mu3nu3.dat'  , c_gxgx_predicted[:,124])

#EOF
