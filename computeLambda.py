#------------------------------------------------------------------------------
#                            computeLambda.py
#------------------------------------------------------------------------------
# This script computes the matrix of correlations C(t), the matrices lambda, M and
# L, and the predicted matrix of correlation with its error. Finally, it computes
# the CG variables taking the matrix lambda as an input, and their errors.
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : July 2017
#                         Modified : September 2017
#                         Run      : python computeLambda.py  (Run with python 2.7)
#------------------------------------------------------------------------------
# There are five theories depending on the selected coarse-grained variables:
# Density of the fluid   : rho
# Energy of the fluid    : e
# x-momentum of the fluid: gx
# z-momentum of the fluid: gz

# Theories:
# Th 1: rho and gz
# Th 2: gx
# Th 3: rho, e, gz
# Th 4: rho, e, gx and gz
# Th 5: e

# It is used the time reversal property and the symmetry of C(0) in order to
# create C(t) and C(0).
# For the system 'fluid' is take into account periodic boundary conditions in
# order to improve the correlations files before creating the matrix C(t).
#           C{mu,nu} = 1/nNodes * sum(C{mu+k,nu+k})
#           k=[0,nNodes-1]

# Onsager reciprocity is used to improve the statistic of matrix M and L.

# The time step in this script is 2e-3, while the time step of lammps was 2e-4.
# This is because the snapshots were saved every 10 time steps.

# In order to compute the matrix lambda, several values of the time 'tau' were
# used. The selected time tau=0.04 gived the lower error (i.e. Frobenious
# norm error of C(t) predicted - C(t)) for the system "solid-fluid", and tau=0.18
#for the the system "fluid".

# Frobeniuos norm error (divided by the number of nodes) is calculated to
# compare C(t) predicted  with C(t).

# Lambda is used to obtain the predicted coarse variables. In order to compare
# the predicted and the original values, it will compute the error as the
# difference between them.
#------------------------------------------------------------------------------

import numpy as np
import datetime
from scipy import linalg
from scipy.linalg import expm
from scipy.linalg import norm

#------------------------------------------------------------------------------
# Screen questions
#------------------------------------------------------------------------------
print 'SYSTEMS'\
'\n' 'f -> fluid' \
'\n' 's -> solid-fluid'
system = raw_input("Select the system you are interested in: ")
print 'CG THEORIES'\
'\n' '1 -> rho and gz' \
'\n' '2 -> gx' \
'\n' '3 -> rho, e and gz' \
'\n' '4 -> rho, e, gx and gz' \
'\n' '5 -> e'
theory         = int(raw_input("Please, select one theory: "))
compute_Ct     = raw_input("Do you want to compute C(t)? (y/n): ")
compute_lambda = raw_input("Do you want to compute lambda, M and L? (y/n): ")
compute_Ctpred = raw_input("Do you want to compute C(t) from lambda? (y/n): ")
compute_CGpred = raw_input("Do you want to compute CG variables from lambda? (y/n): ")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Define global variables
#------------------------------------------------------------------------------
Lx,Ly,Lz    = 40.0,40.0,24.0   #dimensions of the simulation box
nNodes      = 56               #number of nodes
dz          = Lz / nNodes      #bin size
V           = dz * Lx * Ly     #bin volume
dt          = 0.002            #lammps dt=2e-4 (but info saved every 10 steps)
nSteps      = 2000             #t=4 (r.u.). The 'support' of the correlation files after cut them
nStepsModel = 1000             #time to compute the prediction of CG variables
tol         = 1e-3             #rcond in linalg.pinv. It will be use to compute R
tau         = 0.04             #time to which lambda will be calculate
#Number of blocks of C(t) and number of variables
if theory == 1:
    nBlocks = 4
elif theory == 2:
    nBlocks = 1
elif theory == 3:
    nBlocks = 9
elif theory == 4:
    nBlocks = 16
elif theory == 5:
    nBlocks = 1
nVar = int(np.sqrt(nBlocks))
dim  = nVar * nNodes
#Create the matrix epsilon
eps                                = np.identity(dim)
eps[dim-nNodes:dim,dim-nNodes:dim] *= -1
#------------------------------------------------------------------------------
# Define functions
#------------------------------------------------------------------------------
#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(nVar,nVar,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(nVar,nNodes,nVar,nNodes).swapaxes(1,2).ravel()
    return B

#Change format: vector->matrix->matrix.T->vector
def reshape_vmv(A):
    B = (A.reshape(nNodes,nNodes).T).reshape(nNodes**2)
    return B

#Take advantage of the periodic boundary conditions in the creation of C(t) matrix.
# !!! Only for system 'fluid' with periodic boundary conditions.
def pbc(C):
    Cstat = np.zeros((nNodes,nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            for k in range(nNodes):
                Cstat[i,j] += C[(i+k)%nNodes,(j+k)%nNodes]
    return Cstat/nNodes

#Lambda
def computeLambda(Ct,C0):
    row       = tau / dt
    Cforward  = reshape_vm(Ct[row+1,:])
    Cbackward = reshape_vm(Ct[row-1,:])
    Cdev      = (Cforward - Cbackward) / (2 * dt)
    Cdev0     = (reshape_vm(Ct[1,:]) - reshape_vm(Ct[1,:]).T)/ (2 * dt)
    L0        = - Cdev0
    Lanti     = 0.5 * (L0 - L0.T)                                                     #antisymmetric
    L         = 0.5 * (Lanti + eps.dot(Lanti.T).dot(eps))                             #onsager
    Lambda    = - Cdev.dot((linalg.pinv(reshape_vm(Ct[row]), rcond = tol)))
    M0        = Lambda.dot(C0) - L
    M         = 0.5 * (M0 + eps.dot(M0.T).dot(eps))                                   #onsager
    return Lambda, M, L

#Frobenious norm error
def frobenious(A,B):
    error = norm((reshape_vm(A - B)), 'fro') / nNodes
    return error

#C(t) predicted
def computeCtpredict(Ct,C0,Lambda):
    Ctpredict  = np.zeros((nSteps, nNodes**2*nBlocks))
    #Ctdev      = np.zeros((nSteps-1, nNodes**2*nBlocks))
    errorCtpredict = np.zeros(nSteps)
    row = tau / dt
    t   = 0
    for j in range(nSteps):
        print datetime.datetime.now(), 'Computing C(t) predicted. Step', str(j)
        #Ctpredict[j,:]    = reshape_mv(np.dot(expm(-Lambda * t),C0))
        Ctpredict[j,:]    = reshape_mv(np.dot(expm(-Lambda * (t-tau)), reshape_vm(Ct[row])))
        errorCtpredict[j] = frobenious(Ctpredict[j,:],Ct[j,:])
        t+=dt
    #for j in range(nSteps-1):
    #    Cforward          = reshape_vm(Ct[j+1,:])
    #    Cbackward         = reshape_vm(Ct[j-1,:])
    #    Ctdev[j,:]        = reshape_mv((Cforward - Cbackward) / (2 * dt))
    #    t+=dt
    return Ctpredict, errorCtpredict

#CG variables predicted
def computeCGpredict(Lambda,V0):
    Vpred  = np.zeros((nStepsModel, nNodes*nVar))
    t = 0
    for j in range(nStepsModel):
       print datetime.datetime.now(), 'Computing CG variables predicted. Step', str(j)
       Vpred[j,:]    = (np.dot(expm(-Lambda * t),V0)).reshape(nNodes * nVar)
    t+=dt
    return Vpred
#-----------------------------------------------------------------------------------------------------------------------------
#                                                START COMPUTATION
#----------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Compute for rho-gz theory
#------------------------------------------------------------------------------
if theory == 1:
    #-------------
    # Compute C(t)
    #-------------
    if compute_Ct == 'y':
        print datetime.datetime.now(),'Computing C(t) for rho-gz theory...'
        c_rhorho = np.loadtxt('corr_rhorho_2e3steps')
        c_rhogz  = np.loadtxt('corr_rhogz_2e3steps')
        c_gzrho  = np.loadtxt('corr_gzrho_2e3steps')
        c_gzgz   = np.loadtxt('corr_gzgz_2e3steps')
        Ct       = np.zeros((nSteps, nBlocks * nNodes ** 2))

        if system == 's':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack((c_rhorho[i,:], c_rhogz[i,:], c_gzrho[i,:], c_gzgz[i,:])))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))
        if system == 'f':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack(\
                        reshape_mv(pbc(reshape_vm(c_rhorho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhogz[i,:]))),
                        reshape_mv(pbc(reshape_vm(c_gzrho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzgz[i,:])))))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))

        C0      = reshape_vm(Ct[0,:])
        C0_stat = (C0 + C0.T) / 2
        np.savetxt('Ct_2e3steps-rhogzTh', Ct)
        np.savetxt('Ct_100steps-rhogzTh', Ct[0:100,:])
        np.savetxt('C0-rhogzTh', C0_stat)
        print datetime.datetime.now(),'C(t) computed for rho-gz theory!'
    #---------------
    # Compute Lambda
    #---------------
    if compute_lambda == 'y':
        print datetime.datetime.now(),'Computing lambda, M and L for rho-gz theory...'
        Ct         = np.loadtxt('Ct_100steps-rhogzTh')
        C0         = np.loadtxt('C0-rhogzTh')
        Lambda,M,L = computeLambda(Ct,C0)
        np.savetxt('Lambda-rhogzTh', Lambda)
        np.savetxt('M-rhogzTh', M)
        np.savetxt('L-rhogzTh', L)
        print datetime.datetime.now(),'Lambda, M and L computed for rho-gz theory!'
    #-----------------------
    # Compute C(t) predicted
    #-----------------------
    if compute_Ctpred == 'y':
        print datetime.datetime.now(),'Computing C(t) predicted for rho-gz theory...'
        Ct                        = np.loadtxt('Ct_2e3steps-rhogzTh')
        C0                        = np.loadtxt('C0-rhogzTh')
        Lambda                    = np.loadtxt('Lambda-rhogzTh')
        Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
        np.savetxt('Ctpredict-rhogzTh', Ctpredict)
        np.savetxt('errorCtpredict-rhogzTh', errorCtpredict)
        print datetime.datetime.now(),'C(t) predicted computed for rho-gz theory!'
    #-------------------------------
    # Compute CG variables predicted
    #-------------------------------
    if compute_CGpred == 'y':
        print datetime.datetime.now(), 'Computing CG variables for rho-gz theory...'
        Lambda          = np.loadtxt('Lambda-rhogzTh')
        rho             = np.loadtxt('mesoDensity')
        gz              = np.loadtxt('mesoMomentum_z')
        rho0            = rho[0,:]
        gz0             = gz[0,:]
        V0              = np.hstack((rho[0,:], gz[0,:]))
        VPredict        = computeCGpredict(Lambda, V0)
        rhoPredict      = VPredict[:,0:nNodes]
        gzPredict       = VPredict[:,nNodes:2*nNodes]
        errorrhoPredict = rho - rhoPredict
        errorgzPredict  = gz - gzPredict
        np.savetxt('rhoPredict-rhogzTh', rhoPredict)
        np.savetxt('gzPredict-rhogzTh', gzPredict)
        np.savetxt('errorrhoPredict-rhogzTh', errorrhoPredict)
        np.savetxt('errorgzPredict-rhogzTh', errorgzPredict)
        print datetime.datetime.now(),'CG variables computed for gx theory!'

#------------------------------------------------------------------------------
# Compute for gx theory
#------------------------------------------------------------------------------
elif theory == 2:
    #-------------
    # Compute C(t)
    #-------------
    if compute_Ct == 'y':
        print datetime.datetime.now(),'Computing C(t) for gx theory...'
        c_gxgx = np.loadtxt('corr_gxgx_2e3steps')
        Ct     = np.zeros((nSteps, nBlocks * nNodes ** 2))

        if system == 's':
            for i in range(nSteps):
                Ct[i,:] = reshape_mv(((reshape_vm(c_gxgx[i,:]) + reshape_vm(c_gxgx[i,:]).T) / 2))
        elif system == 'f':
            for i in range(nSteps):
                Ct[i,:] = reshape_mv(((pbc(reshape_vm(c_gxgx[i,:])) + pbc(reshape_vm(c_gxgx[i,:]).T)) / 2))

        C0      = reshape_vm(Ct[0,:])
        C0_stat = (C0 + C0.T) / 2
        np.savetxt('Ct_2e3steps-gxTh', Ct)
        np.savetxt('Ct_100steps-gxTh', Ct[0:100,:])
        np.savetxt('C0-gxTh', C0_stat)
        print datetime.datetime.now(),'C(t) computed for gx theory!'
    #---------------
    # Compute Lambda
    #---------------
    if compute_lambda == 'y':
        print datetime.datetime.now(),'Computing lambda, M and L for gx theory...'
        #Ct         = np.loadtxt('Ct_100steps-gxTh')
        Ct         = np.loadtxt('Ct_100steps-gxTh')
        C0         = np.loadtxt('C0-gxTh')
        #Lambda,M,L = computeLambda(Ct,C0)
        #np.savetxt('Lambda-gxTh', Lambda)
        #np.savetxt('M-gxTh', M)
        #np.savetxt('L-gxTh', L)
        for tau in np.arange(0.15,0.19,0.01):
            Lambda,M,L = computeLambda(Ct,C0)
            np.savetxt('Lambda-gxTh-tau'+str(tau), Lambda)
            np.savetxt('M-gxTh-tau'+str(tau), M)
            np.savetxt('L-gxTh-tau'+str(tau), L)
        print datetime.datetime.now(),'Lambda, M and L computed for gx theory!'
    #-----------------------
    # Compute C(t) predicted
    #-----------------------
    if compute_Ctpred == 'y':
        print datetime.datetime.now(),'Computing C(t) predicted for gx theory...'
        Ct                        = np.loadtxt('Ct_2e3steps-gxTh')
        C0                        = np.loadtxt('C0-gxTh')
        #Lambda                    = np.loadtxt('Lambda-gxTh')
        #Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
        #np.savetxt('Ctpredict-gxTh', Ctpredict)
        #np.savetxt('errorCtpredict-gxTh', errorCtpredict)
        for tau in np.arange(0.15,0.19,0.01):
            Lambda                    = np.loadtxt('Lambda-gxTh-tau'+str(tau))
            Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
            np.savetxt('Ctpredict-gxTh-tau'+str(tau), Ctpredict)
            np.savetxt('errorCtpredict-gxTh-tau'+str(tau), errorCtpredict)
        print datetime.datetime.now(),'C(t) predicted computed for gx theory!'
    #-------------------------------
    # Compute CG variables predicted
    #-------------------------------
    if compute_CGpred == 'y':
        print datetime.datetime.now(), 'Computing CG variables for gx theory...'
        Lambda         = np.loadtxt('Lambda-gxTh')
        gx             = np.loadtxt('mesoMomentum_x')
        gx0            = gx[0,:]
        gxPredict      = computeCGpredict(Lambda, gx0)
        errorgxPredict = gx - gxPredict
        np.savetxt('gxPredict-gxTheory', gxPredict)
        np.savetxt('errorgxPredict-gxTheory', errorgxPredict)
        print datetime.datetime.now(),'CG variables computed for gx theory!'

#------------------------------------------------------------------------------
# Compute for rho-e-gz theory
#------------------------------------------------------------------------------
elif theory == 3:
    #-------------
    # Compute C(t)
    #-------------
    if compute_Ct == 'y':
        print datetime.datetime.now(),'Computing C(t) for rho-e-gz theory...'
        c_rhorho = np.loadtxt('corr_rhorho_2e3steps')
        c_rhoe   = np.loadtxt('corr_rhoe_2e3steps')
        c_rhogz  = np.loadtxt('corr_rhogz_2e3steps')
        c_erho   = np.loadtxt('corr_erho_2e3steps')
        c_ee     = np.loadtxt('corr_ee_2e3steps')
        c_egz    = np.loadtxt('corr_egz_2e3steps')
        c_gzrho  = np.loadtxt('corr_gzrho_2e3steps')
        c_gze    = np.loadtxt('corr_gze_2e3steps')
        c_gzgz   = np.loadtxt('corr_gzgz_2e3steps')
        Ct       = np.zeros((nSteps, nBlocks * nNodes ** 2))

        if system == 's':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack((\
                        c_rhorho[i,:], c_rhoe[i,:], c_rhogz[i,:],\
                        c_erho[i,:],   c_ee[i,:],   c_egz[i,:],
                        c_gzrho[i,:],  c_gze[i,:],  c_gzgz[i,:])))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))
        if system == 'f':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack(\
                        reshape_mv(pbc(reshape_vm(c_rhorho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhoe[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhogz[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_erho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_ee[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_egz[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzrho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gze[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzgz[i,:])))))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))

        C0      = reshape_vm(Ct[0,:])
        C0_stat = (C0 + C0.T) / 2
        np.savetxt('Ct_2e3steps-rhoegzTh', Ct)
        np.savetxt('Ct_100steps-rhoegzTh', Ct[0:100,:])
        np.savetxt('C0-rhoegzTh', C0_stat)
        print datetime.datetime.now(),'C(t) computed for rho-e-gz theory!'
    #---------------
    # Compute Lambda
    #---------------
    if compute_lambda == 'y':
        print datetime.datetime.now(),'Computing lambda, M and L for rho-e-gz theory...'
        Ct         = np.loadtxt('Ct_100steps-rhoegzTh')
        C0         = np.loadtxt('C0-rhoegzTh')
        Lambda,M,L = computeLambda(Ct,C0)
        np.savetxt('Lambda-rhoegzTh', Lambda)
        np.savetxt('M-rhoegzTh', M)
        np.savetxt('L-rhoegzTh', L)
        print datetime.datetime.now(),'Lambda, M and L computed for rho-e-gz theory!'
    #-----------------------
    # Compute C(t) predicted
    #-----------------------
    if compute_Ctpred == 'y':
        print datetime.datetime.now(),'Computing C(t) predicted for rho-e-gz theory...'
        Ct                        = np.loadtxt('Ct_2e3steps-rhoegzTh')
        C0                        = np.loadtxt('C0-rhoegzTh')
        Lambda                    = np.loadtxt('Lambda-rhoegzTh')
        Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
        np.savetxt('Ctpredict-rhoegzTh', Ctpredict)
        np.savetxt('errorCtpredict-rhoegzTh', errorCtpredict)
        print datetime.datetime.now(),'C(t) predicted computed for rho-e-gz theory!'
    #-------------------------------
    # Compute CG variables predicted
    #-------------------------------
    if compute_CGpred == 'y':
        print datetime.datetime.now(), 'Computing CG variables for rho-e-gz theory...'
        Lambda          = np.loadtxt('Lambda-rhoegzTh')
        rho             = np.loadtxt('mesoDensity')
        e               = np.loadtxt('mesoEnergy')
        gz              = np.loadtxt('mesoMomentum_z')
        V0              = np.hstack((rho[0,:], e[0,:], gz[0,:]))
        VPredict        = computeCGpredict(Lambda, V0)
        rhoPredict      = VPredict[:,0:nNodes]
        ePredict        = VPredict[:,nNodes:2*nNodes]
        gzPredict       = VPredict[:,2*nNodes:3*nNodes]
        errorrhoPredict = rho - rhoPredict
        errorePredict   = e - ePredict
        errorgzPredict  = gz - gzPredict
        np.savetxt('rhoPredict-rhoegzTh', rhoPredict)
        np.savetxt('ePredict-rhoegzTh', ePredict)
        np.savetxt('gzPredict-rhoegzTh', gzPredict)
        np.savetxt('errorrhoPredict-rhoegzTh', errorrhoPredict)
        np.savetxt('errorePredict-rhoegzTh', errorePredict)
        np.savetxt('errorgzPredict-rhoegzTh', errorgzPredict)
        print datetime.datetime.now(),'CG variables computed for rho-e-gz theory!'

#------------------------------------------------------------------------------
# Compute for rho-e-gx-gz theory
#------------------------------------------------------------------------------
elif theory == 4:
    #-------------
    # Compute C(t)
    #-------------
    if compute_Ct == 'y':
        print datetime.datetime.now(),'Computing C(t) for rho, e, gx and gz theory...'
        c_rhorho = np.loadtxt('corr_rhorho_2e3steps')
        c_rhoe   = np.loadtxt('corr_rhoe_2e3steps')
        c_rhogx  = np.loadtxt('corr_rhogx_2e3steps')
        c_rhogz  = np.loadtxt('corr_rhogz_2e3steps')
        c_erho   = np.loadtxt('corr_erho_2e3steps')
        c_ee     = np.loadtxt('corr_ee_2e3steps')
        c_egx    = np.loadtxt('corr_egx_2e3steps')
        c_egz    = np.loadtxt('corr_egz_2e3steps')
        c_gxrho  = np.loadtxt('corr_gxrho_2e3steps')
        c_gxe    = np.loadtxt('corr_gxe_2e3steps')
        c_gxgx   = np.loadtxt('corr_gxgx_2e3steps')
        c_gxgz   = np.loadtxt('corr_gxgz_2e3steps')
        c_gzrho  = np.loadtxt('corr_gzrho_2e3steps')
        c_gze    = np.loadtxt('corr_gze_2e3steps')
        c_gzgx   = np.loadtxt('corr_gzgx_2e3steps')
        c_gzgz   = np.loadtxt('corr_gzgz_2e3steps')
        Ct       = np.zeros((nSteps, nBlocks * nNodes ** 2))

        if system == 's':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack((\
                        c_rhorho[i,:], c_rhoe[i,:], c_rhogx[i,:], c_rhogz[i,:],\
                        c_erho[i,:],   c_ee[i,:],   c_egx[i,:],   c_egz[i,:],\
                        c_gxrho[i,:],  c_gxe[i,:],  c_gxgx[i,:],  c_gxgz[i,:], \
                        c_gzrho[i,:],  c_gze[i,:],  c_gzgx[i,:],  c_gzgz[i,:])))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))
        if system == 'f':
            for i in range(nSteps):
                Ct1     = reshape_vm(np.hstack(\
                        reshape_mv(pbc(reshape_vm(c_rhorho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhoe[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhogx[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_rhogz[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_erho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_ee[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_egx[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_egz[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gxrho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gxe[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gxgx[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gxgz[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzrho[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gze[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzgx[i,:]))),\
                        reshape_mv(pbc(reshape_vm(c_gzgz[i,:])))))
                Ct[i,:] = reshape_mv(0.5 * (Ct1+eps.dot(Ct1.T).dot(eps)))

        C0      = reshape_vm(Ct[0,:])
        C0_stat = (C0 + C0.T) / 2
        np.savetxt('Ct_2e3steps-rhoegxgzTh', Ct)
        np.savetxt('Ct_100steps-rhoegxgzTh', Ct[0:100,:])
        np.savetxt('C0-rhoegxgzTh', C0_stat)
        print datetime.datetime.now(),'C(t) computed for rho-e-gx-gz theory!'
    #---------------
    # Compute Lambda
    #---------------
    if compute_lambda == 'y':
        print datetime.datetime.now(),'Computing lambda, M and L for rho, e, gx and gz theory...'
        Ct         = np.loadtxt('Ct_100steps-rhoegxgzTh')
        C0         = np.loadtxt('C0-rhoegxgzTh')
        Lambda,M,L = computeLambda(Ct,C0)
        np.savetxt('Lambda-rhoegxgzTh', Lambda)
        np.savetxt('M-rhoegxgzTh', M)
        np.savetxt('L-rhoegxgzTh', L)
        print datetime.datetime.now(),'Lambda, M and L computed for rho-e-gx-gz theory!'
    #-----------------------
    # Compute C(t) predicted
    #-----------------------
    if compute_Ctpred == 'y':
        print datetime.datetime.now(),'Computing C(t) predicted for rho-e-gx-gz theory...'
        Ct                        = np.loadtxt('Ct_2e3steps-rhoegxgzTh')
        C0                        = np.loadtxt('C0-rhoegxgzTh')
        Lambda                    = np.loadtxt('Lambda-rhoegxgzTh')
        Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
        np.savetxt('Ctpredict-rhoegxgzTh', Ctpredict)
        np.savetxt('errorCtpredict-rhoegxgzTh', errorCtpredict)
        print datetime.datetime.now(),'C(t) predicted computed for rho-e-gx-gz theory!'
    #-------------------------------
    # Compute CG variables predicted
    #-------------------------------
    if compute_CGpred == 'y':
        print datetime.datetime.now(), 'Computing CG variables for rho-e-gx-gz theory...'
        Lambda          = np.loadtxt('Lambda-rhoegxgzTh')
        rho             = np.loadtxt('mesoDensity')
        e               = np.loadtxt('mesoEnergy')
        gx              = np.loadtxt('mesoMomentum_x')
        gz              = np.loadtxt('mesoMomentum_z')
        V0              = np.hstack((rho[0,:], e[0,:], gx[0,:], gz[0,:]))
        VPredict        = computeCGpredict(Lambda, V0)
        rhoPredict      = VPredict[:,0:nNodes]
        ePredict        = VPredict[:,nNodes:2*nNodes]
        gxPredict       = VPredict[:,2*nNodes:3*nNodes]
        gzPredict       = VPredict[:,3*nNodes:4*nNodes]
        errorrhoPredict = rho - rhoPredict
        errorePredict   = e - ePredict
        errorgxPredict  = gx - gxPredict
        errorgzPredict  = gz - gzPredict
        np.savetxt('rhoPredict-rhoegxgzTh', rhoPredict)
        np.savetxt('ePredict-rhoegxgzTh', ePredict)
        np.savetxt('gxPredict-rhoegxgzTh', gxPredict)
        np.savetxt('gzPredict-rhoegxgzTh', gzPredict)
        np.savetxt('errorrhoPredict-rhoegxgzTh', errorrhoPredict)
        np.savetxt('errorePredict-rhoegxgzTh', errorePredict)
        np.savetxt('errorgxPredict-rhoegxgzTh', errorgxPredict)
        np.savetxt('errorgzPredict-rhoegxgzTh', errorgzPredict)
        print datetime.datetime.now(),'CG variables computed for rho-e-gx-gz theory!'

#------------------------------------------------------------------------------
# Compute for e theory
#------------------------------------------------------------------------------
elif theory == 5:
    #-------------
    # Compute C(t)
    #-------------
    if compute_Ct == 'y':
        print datetime.datetime.now(),'Computing C(t) for e theory...'
        c_ee = np.loadtxt('corr_ee_2e3steps')
        Ct   = np.zeros((nSteps, nBlocks * nNodes ** 2))

        if system == 's':
            for i in range(nSteps):
                Ct[i,:] = reshape_mv(((reshape_vm(c_ee[i,:]) + reshape_vm(c_ee[i,:]).T) / 2))
        elif system == 'f':
            for i in range(nSteps):
                Ct[i,:] = reshape_mv(((pbc(reshape_vm(c_ee[i,:])) + pbc(reshape_vm(c_ee[i,:]).T)) / 2))

        C0      = reshape_vm(Ct[0,:])
        C0_stat = (C0 + C0.T) / 2
        np.savetxt('Ct_2e3steps-eTh', Ct)
        np.savetxt('Ct_100steps-eTh', Ct[0:100,:])
        np.savetxt('C0-eTh', C0_stat)
        print datetime.datetime.now(),'C(t) computed for e theory!'
    #---------------
    # Compute Lambda
    #---------------
    if compute_lambda == 'y':
        print datetime.datetime.now(),'Computing lambda, M and L for e theory...'
        Ct         = np.loadtxt('Ct_100steps-eTh')
        C0         = np.loadtxt('C0-eTh')
        Lambda,M,L = computeLambda(Ct,C0)
        np.savetxt('Lambda-eTh', Lambda)
        np.savetxt('M-eTh', M)
        np.savetxt('L-eTh', L)
        print datetime.datetime.now(),'Lambda, M and L computed for e theory!'
    #-----------------------
    # Compute C(t) predicted
    #-----------------------
    if compute_Ctpred == 'y':
        print datetime.datetime.now(),'Computing C(t) predicted for e theory...'
        Ct                        = np.loadtxt('Ct_2e3steps-eTh')
        C0                        = np.loadtxt('C0-eTh')
        Lambda                    = np.loadtxt('Lambda-eTh')
        Ctpredict, errorCtpredict = computeCtpredict(Ct,C0,Lambda)
        np.savetxt('Ctpredict-eTh', Ctpredict)
        np.savetxt('errorCtpredict-eTh', errorCtpredict)
        print datetime.datetime.now(),'C(t) predicted computed for e theory!'
    #-------------------------------
    # Compute CG variables predicted
    #-------------------------------
    if compute_CGpred == 'y':
        print datetime.datetime.now(), 'Computing CG variables for e theory...'
        Lambda   = np.loadtxt('Lambda-eTh')
        e        = np.loadtxt('mesoEnergy')
        e0       = e[0,:]
        ePredict = computeCGpredict(Lambda, e0)
        errorePredict = e - ePredict
        np.savetxt('ePredict-eTh', ePredict)
        np.savetxt('errorePredict-eTh', errorePredict)
        print datetime.datetime.now(),'CG variables computed for e theory!'

#----------------------------------------------------------------------------------------------------------------------------
#                                                END COMPUTATION
#-----------------------------------------------------------------------------------------------------------------------------
