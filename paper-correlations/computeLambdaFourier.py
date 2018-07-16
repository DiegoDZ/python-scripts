#---------------------------------------------------------
#                   computeLambdaFourier.py
#---------------------------------------------------------
# This script takes the antidiagonal of selected frames
# for different time dependent matrices. It uses two ways
# to obtain it: based on real and fourier space.
#           Author: @DiegoDZ
#           Date:   oct 2017
#           Run:    >>> python computeLambdaFourier.py
#---------------------------------------------------------

import numpy as np
#import matplotlib.pyplot as plt
from scipy import linalg

#--------------------------------------Screen question-----------------------------
space  = raw_input("Fourier or real space? (f/r): ")
if space == 'f':
    label = 'Fourier'
elif space == 'r':
    label = 'Real'

#-------------------------------------------Input---------------------------------
#Ctinput            = np.loadtxt('Ct-gxTh')                                        #matrix of correlations
#CtdevInput         = np.loadtxt('Ctdev-gxTh')                                     #derivative of matrix of correlations
Ctinput            = np.loadtxt('Ct-gxTh-500steps')                              #matrix of correlations
CtdevInput         = np.loadtxt('Ctdev-gxTh-500steps')                           #derivative of matrix of correlations
#Eta                = np.loadtxt('viscosityt')
tstart,tstop,tdump = 0.0,0.21,0.01                                                 #times in which the info will be save
Lx,Ly,Lz           = 40.0,40.0,30.0                                                #dimensions of the simulation box
nNodes             = 60                                                            #number of nodes
dz                 = Lz / nNodes                                                   #bin size
nBlocks            = 1                                                             #number blocks of matrix of correlations
nVar               = int(np.sqrt(nBlocks))                                         #number variables
dim                = nVar * nNodes                                                 #dimension matrix of correlations
dt                 = 0.002                                                         #time step
tol                = 1e-3                                                          #rcond in linalg.pinv. It will be use to compute Cinv
#nSteps             = 15000
nSteps             = 500

E                  = np.zeros((nNodes, nNodes), dtype = complex)                   #unitary matrix. It will be used for Fourier transforms
Einv               = np.zeros((nNodes, nNodes), dtype = complex)
for mu in range(nNodes):
    for nu in range(nNodes):
            E[mu,nu]    = np.exp( 1j*2*np.pi*mu*nu/nNodes)/np.sqrt(nNodes)
            Einv[mu,nu] = np.exp(-1j*2*np.pi*mu*nu/nNodes)/np.sqrt(nNodes)

laplacian = (1 / dz**2) * (-2 * (np.eye(nNodes)) +\
          np.eye(nNodes, nNodes, -nNodes+1) + np.eye(nNodes, nNodes, nNodes-1) +
          np.eye(nNodes, nNodes, -1) + np.eye(nNodes, nNodes, 1))
D = Einv.dot(laplacian).dot(E)                                                    #laplacian in fourier space

#--------------------------------------Subrutines-----------------------------
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(nVar,nVar,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B
def reshape_mv(A):
    B = A.reshape(nVar,nNodes,nVar,nNodes).swapaxes(1,2).ravel()
    return B
def row(t):
    row = int(round(t/dt))
    return row     #-1 because the index of the first row is 0
def avgDiag(A):
    B = np.zeros(2*nNodes-1)
    for i in np.arange(-(nNodes-1),nNodes,1):
        B[i+nNodes-1] = np.sum(np.diag(A,i))/np.shape(np.diag(A,i))
    return B

P = np.eye(nNodes)
for i in np.arange(1,15,1):
    P +=  np.eye(nNodes, nNodes, -nNodes+i) + np.eye(nNodes, nNodes, nNodes-i) + np.eye(nNodes, nNodes, -i) + np.eye(nNodes, nNodes, i)
def pulir(A):
    B = (A + 2.5e-5)*P-2.5e-5
    return B

#------------------------------------------------------------------------------------------------
#                                   Start computation
#------------------------------------------------------------------------------------------------

#------------------Real space -----------------------

if space == 'r':
    CtdevReal = np.zeros((nSteps, nNodes**2))
    CtinvReal = np.zeros((nSteps, nNodes**2))
    for i in np.arange(1, nSteps-1, 1):
        Ctforward  = reshape_vm(Ctinput[i+1,:])
        Ctbackward = reshape_vm(Ctinput[i-1,:])
        CtdevReal[i]   = reshape_mv((Ctforward - Ctbackward) / (2 * dt))
    for i in np.arange(0,nSteps,1):
        CtinvReal[i] = reshape_mv(linalg.pinv(reshape_vm(Ctinput[i]), rcond = tol))
    #Average over all anti-diagonals for different times
    for t in np.arange(tstart, tstop, tdump):
        Ct      = pulir(reshape_vm(Ctinput[row(t),:]))
        Ct2      = reshape_vm(Ctinput[row(t),:])
        #Ct      = avgDiag(pulir(reshape_vm(Ctinput[row(t),:])))
        Ctdev   = avgDiag(reshape_vm(CtdevReal[row(t),:]))
        Ctinv   = avgDiag(reshape_vm(CtinvReal[row(t),:]))
        Lambda  = avgDiag(-reshape_vm(CtdevReal[row(t),:]).dot(reshape_vm(CtinvReal[row(t),:])))
        kinVisc = avgDiag(-reshape_vm(CtdevReal[row(t),:]).dot(reshape_vm(CtinvReal[row(t),:])).dot(linalg.pinv(laplacian, rcond=tol)))
        CtCtinv0 = Ct.dot(reshape_vm(CtinvReal[row(0),:]))
        np.savetxt('C-gxTh-tau'+str(t), Ct2)
        np.savetxt('PULIDO-C-gxTh-tau'+str(t), Ct)
        #np.savetxt('PULIDO-Ct_2e3steps-gxTh-tau'+str(t), Ct)
        #np.savetxt('Ctdev_2e3steps-gxTh-tau'+str(t), Ctdev)
        #np.savetxt('Ctinv_2e3steps-gxTh-tau'+str(t), Ctinv)
        #np.savetxt('Lambda_2e3steps-gxTh-tau'+str(t), Lambda)
        #np.savetxt('kinVisc_2e3steps-gxTh-tau'+str(t), kinVisc)

#------------------Fourier space ---------------------

if space == 'f':
    CtFourier      = np.zeros((nSteps, nNodes**2))
    CtFourierDiag  = np.zeros((nSteps, nNodes))
    EtaFourier     = np.zeros((nSteps, nNodes**2))
    EtaFourierDiag = np.zeros((nSteps, nNodes))
    CtdevFourier   = np.zeros((nSteps, nNodes**2))
    CtinvFourier   = np.zeros((nSteps, nNodes**2))
    CtCtinv0Fourier = np.zeros((nSteps, nNodes**2))
    
    for i in range(nSteps):
        #print i
        CtFourier[i,:]       = reshape_mv(Einv.dot(pulir(reshape_vm(Ctinput[i,:]))).dot(E))
        CtdevFourier[i,:]    = reshape_mv(Einv.dot(reshape_vm(CtdevInput[i,:])).dot(E))
        CtFourierDiag[i,:]   = np.diag(Einv.dot(pulir(reshape_vm(Ctinput[i,:]))).dot(E))
        #EtaFourier[i,:]      = reshape_mv(Einv.dot(reshape_vm(Eta[i,:])).dot(E))
        #EtaFourierDiag[i,:]  = np.diag(Einv.dot(reshape_vm(Eta[i,:])).dot(E))
        CtinvFourier[i,:]    = reshape_mv(linalg.pinv(reshape_vm(CtFourier[i,:]), rcond=tol))
    #np.savetxt('CtFourier', CtFourier)
    #np.savetxt('viscositytFourier', EtaFourier)
    #np.savetxt('CtFourierDiag', CtFourierDiag)
    #np.savetxt('viscositytFourierDiag', EtaFourierDiag)
    #for i in np.arange(1, nSteps-1, 1):
    #    print i 
    #    Ctforward         = reshape_vm(CtFourier[i+1,:])
    #    Ctbackward        = reshape_vm(CtFourier[i-1,:])
    #    CtdevFourier[i,:] = reshape_mv((Ctforward - Ctbackward) / (2*dt))
    #CtdevFourier0     = (reshape_vm(CtFourier[1,:])-reshape_vm(CtFourier[1,:].T)) / (2*dt)  #Zero because it is symmetric
    #CtdevFourier[0,:] = reshape_mv(CtdevFourier0)
    #np.savetxt('CtdevFourier-test2', reshape_vm(CtdevFourier[0,:]))
    #Take the diagonal for different times
    #for t in np.arange(tstart, tstop, tdump):
    #    Ct            = np.diag(reshape_vm(CtFourier[row(t),:]))
    #    Ctdev         = np.diag(reshape_vm(CtdevFourier[row(t),:]))
    #    Ctinv         = np.diag(reshape_vm(CtinvFourier[row(t),:]))
    #    Lambda        = np.diag(-reshape_vm(CtdevFourier[row(t),:]).dot(reshape_vm(CtinvFourier[row(t),:])))
    #    kinVisc       = -Lambda[1:nNodes]/np.diag(D.real)[1:nNodes]
    #    CtCtinv0      = np.diag(reshape_vm(CtCtinv0Fourier[row(t),:]))
    #    #np.savetxt('PULIDO-CtFourier_2e3steps-f-gxTh-tau'+str(t), Ct)
    #    #np.savetxt('PULIDO-CtdevFourier_2e3steps-f-gxTh-tau'+str(t), Ctdev[1:nNodes])
    #    #np.savetxt('PULIDO-CtinvFourier_2e3steps-f-gxTh-tau'+str(t), Ctinv[1:nNodes])
    #    #np.savetxt('PULIDO-LambdaFourier-2e3steps-f-gxTh-tau'+str(t), Lambda[1:nNodes])
    #    #np.savetxt('PULIDO-kinViscFourier_2e3steps-f-gxTh-tau'+str(t), kinVisc)
    #    #np.savetxt('PULIDO-CtCtinv0Fourier_2e3steps-f-gxTh-tau'+str(t), CtCtinv0)
    #t = 0
    #Ct      = np.diag(reshape_vm(CtFourier[row(t),:]))
    #np.savetxt('CtFourier_2e3steps-f-gxTh-t'+str(t), Ct)

    #Calculate the evolution of the modes of Lambda
    Lambdat = np.zeros((nSteps-1, nNodes))
    for step in range(nSteps-1):
        print step
        Lambda          = np.diag(-reshape_vm(CtdevFourier[step,:]).dot(reshape_vm(CtinvFourier[step,:])))
        Lambdat[step,:] = Lambda
    #Save files
    for k in np.arange(1,nNodes+1,1):
        np.savetxt('LambdaFourier-k'+str(k)+'.dat',  Lambdat[:,k-1])
    print CtdevFourier[0,:]
    
#EOF
