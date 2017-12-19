#---------------------------------------------------------
#                   computeLambdaFourier.py
#---------------------------------------------------------
# This script takes the antidiagonal of selected frames
# for different time dependent matrices. It uses two ways
# to obtain it: based on real and fourier space.
#           Author: @DiegoDZ
#           Date:   October 2017
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
Ctinput            = np.loadtxt('Ct_2e3steps-gxTh')                                        #matrix of correlations
tstart,tstop,tdump = 0.1,0.21,0.01                                                         #times in which the info will be save
Lx,Ly,Lz           = 40.0,40.0,30.0                                                        #dimensions of the simulation box
nNodes             = 60                                                                    #number of nodes
dz                 = Lz / nNodes                                                           #bin size
nBlocks            = 1                                                                     #number blocks of matrix of correlations
nVar               = int(np.sqrt(nBlocks))                                                 #number variables
dim                = nVar * nNodes                                                         #dimension matrix of correlations
dt                 = 0.002                                                                 #time step
tol                = 1e-3                                                                  #rcond in linalg.pinv. It will be use to compute Cinv
nSteps             = 2000
E                  = np.zeros((nNodes, nNodes), dtype = complex)                           #unitary matrix. It will be used for Fourier transforms
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
        np.savetxt('C-gxTh-tau'+str(t), Ct2)
        np.savetxt('PULIDO-C-gxTh-tau'+str(t), Ct)
        #np.savetxt('PULIDO-Ct_2e3steps-gxTh-tau'+str(t), Ct)
        #np.savetxt('Ctdev_2e3steps-gxTh-tau'+str(t), Ctdev)
        #np.savetxt('Ctinv_2e3steps-gxTh-tau'+str(t), Ctinv)
        #np.savetxt('Lambda_2e3steps-gxTh-tau'+str(t), Lambda)
        #np.savetxt('kinVisc_2e3steps-gxTh-tau'+str(t), kinVisc)

#------------------Fourier space ---------------------

if space == 'f':
    CtFourier    = np.zeros((nSteps, nNodes**2))
    CtdevFourier = np.zeros((nSteps, nNodes**2))
    CtinvFourier = np.zeros((nSteps, nNodes**2))
    for i in range(nSteps):
        CtFourier[i,:]    = reshape_mv(Einv.dot(pulir(reshape_vm(Ctinput[i,:]))).dot(E))
        #CtFourier[i,:]    = reshape_mv(Einv.dot(reshape_vm(Ctinput[i,:])).dot(E))
        CtinvFourier[i,:] = reshape_mv(linalg.pinv(reshape_vm(CtFourier[i,:]), rcond=tol))
    for i in np.arange(1, nSteps-1, 1):
        Ctforward       = reshape_vm(CtFourier[i+1,:])
        Ctbackward      = reshape_vm(CtFourier[i-1,:])
        CtdevFourier[i] = reshape_mv((Ctforward - Ctbackward) / (2 * dt))
    #Take the diagonal for different times
    for t in np.arange(tstart, tstop, tdump):
        Ct      = np.diag(reshape_vm(CtFourier[row(t),:]))
        Ctdev   = np.diag(reshape_vm(CtdevFourier[row(t),:]))
        Ctinv   = np.diag(reshape_vm(CtinvFourier[row(t),:]))
        Lambda  = np.diag(-reshape_vm(CtdevFourier[row(t),:]).dot(reshape_vm(CtinvFourier[row(t),:])))
        kinVisc = -Lambda[1:nNodes]/np.diag(D.real)[1:nNodes]
        np.savetxt('PULIDO-CtFourier_2e3steps-f-gxTh-tau'+str(t), Ct)
        np.savetxt('PULIDO-CtdevFourier_2e3steps-f-gxTh-tau'+str(t), Ctdev[1:nNodes])
        np.savetxt('PULIDO-CtinvFourier_2e3steps-f-gxTh-tau'+str(t), Ctinv[1:nNodes])
        np.savetxt('PULIDO-LambdaFourier_2e3steps-f-gxTh-tau'+str(t), Lambda[1:nNodes])
        np.savetxt('PULIDO-kinViscFourier_2e3steps-f-gxTh-tau'+str(t), kinVisc)
    #t = 0
    #Ct      = np.diag(reshape_vm(CtFourier[row(t),:]))
    #np.savetxt('CtFourier_2e3steps-f-gxTh-t'+str(t), Ct)

"""
#------------------------------------------------------------------------------------------------
#                                   Figures
#------------------------------------------------------------------------------------------------
#Plot results - fourier space
#Load files
Ct0        = np.loadtxt('Ct_2e3steps-gxTh-t0')
Ct02       = np.loadtxt('Ct_2e3steps-gxTh-t0.2')
Ct04       = np.loadtxt('Ct_2e3steps-gxTh-t0.4')
Ct2        = np.loadtxt('Ct_2e3steps-gxTh-t2')
Ct5        = np.loadtxt('Ct_2e3steps-gxTh-t5')
Ct10       = np.loadtxt('Ct_2e3steps-gxTh-t10')
Ct15       = np.loadtxt('Ct_2e3steps-gxTh-t15')

Ctdev02    = np.loadtxt('Ctdev_2e3steps-gxTh-t0.2')
Ctdev04    = np.loadtxt('Ctdev_2e3steps-gxTh-t0.4')
Ctdev2     = np.loadtxt('Ctdev_2e3steps-gxTh-t2')
Ctdev5     = np.loadtxt('Ctdev_2e3steps-gxTh-t5')
Ctdev15    = np.loadtxt('Ctdev_2e3steps-gxTh-t15')

Ctinv02    = np.loadtxt('Ctinv_2e3steps-gxTh-t0.2')
Ctinv04    = np.loadtxt('Ctinv_2e3steps-gxTh-t0.4')
Ctinv5     = np.loadtxt('Ctinv_2e3steps-gxTh-t5')
Ctinv15    = np.loadtxt('Ctinv_2e3steps-gxTh-t15')

Lambdat02  = np.loadtxt('Lambda_2e3steps-gxTh-t15')
Lambdat04  = np.loadtxt('Lambda_2e3steps-gxTh-t0')
Lambdat5   = np.loadtxt('Lambda_2e3steps-gxTh-t0.4')
Lambdat15  = np.loadtxt('Lambda_2e3steps-gxTh-t15')

kinVisct02 = np.loadtxt('kinVisc_2e3steps-gxTh-t0.2')
kinVisct04 = np.loadtxt('kinVisc_2e3steps-gxTh-t0.4')
kinVisct5  = np.loadtxt('kinVisc_2e3steps-gxTh-t5')
kinVisct15 = np.loadtxt('kinVisc_2e3steps-gxTh-t15')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ct0, linestyle='-', color='k')
plt.hold('on')
plt.plot(Ct02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ct04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ct2, linestyle='-', color='m')
plt.hold('on')
plt.plot(Ct5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ct10, linestyle='-', color='c')
plt.hold('on')
plt.plot(Ct15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C(t=0)$', '$C(t=0.2)$', '$C(t=0.4)$', '$C(t=2)$', '$C(t=5)$', '$C(t=10)$', '$C(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$C(t)$')
plt.savefig('CtFrames-56n.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ctdev02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctdev04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctdev5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ctdev15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$\dot{C}(t=0.2)$', '$\dot{C}(t=0.4)$', '$\dot{C}(t=5)$', '$\dot{C}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$\dot{C}(t)$')
plt.savefig('CtdevFrames-56n.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim((-4e-5, 3e-5))
ax.grid()
plt.plot(Ctdev02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctdev04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctdev5, linestyle='-', color='grey')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$\dot{C}(t=0.2)$', '$\dot{C}(t=0.4)$', '$\dot{C}(t=5)$', '$\dot{C}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$\dot{C}(t)$')
plt.savefig('CtdevFrames-56n-zoom.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ctinv02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctinv04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctinv5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ctinv15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C^{-1}(t=0.2)$', '$C^{-1}(t=0.4)$', '$C^{-1}(t=5)$', '$C^{-1}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$C^{-1}(t)$')
plt.savefig('CtinvFrames-56n.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim((-5e4, 1e5))
ax.grid()
plt.plot(Ctinv02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctinv04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctinv5, linestyle='-', color='grey')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C^{-1}(t=0.2)$', '$C^{-1}(t=0.4)$', '$C^{-1}(t=5)$', '$C^{-1}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$C^{-1}(t)$')
plt.savefig('CtinvFrames-56n-zoom.eps')

#Plot results - fourier space
#Load files
Ct0        = np.loadtxt('CtFourier_2e3steps-gxTh-t0')
Ct02       = np.loadtxt('CtFourier_2e3steps-gxTh-t0.2')
Ct04       = np.loadtxt('CtFourier_2e3steps-gxTh-t0.4')
Ct2        = np.loadtxt('CtFourier_2e3steps-gxTh-t2')
Ct5        = np.loadtxt('CtFourier_2e3steps-gxTh-t5')
Ct10       = np.loadtxt('CtFourier_2e3steps-gxTh-t10')
Ct15       = np.loadtxt('CtFourier_2e3steps-gxTh-t15')

Ctdev02    = np.loadtxt('CtdevFourier_2e3steps-gxTh-t0.2')
Ctdev04    = np.loadtxt('CtdevFourier_2e3steps-gxTh-t0.4')
Ctdev2     = np.loadtxt('CtdevFourier_2e3steps-gxTh-t2')
Ctdev5     = np.loadtxt('CtdevFourier_2e3steps-gxTh-t5')
Ctdev15    = np.loadtxt('CtdevFourier_2e3steps-gxTh-t15')

Ctinv02    = np.loadtxt('CtinvFourier_2e3steps-gxTh-t0.2')
Ctinv04    = np.loadtxt('CtinvFourier_2e3steps-gxTh-t0.4')
Ctinv5     = np.loadtxt('CtinvFourier_2e3steps-gxTh-t5')
Ctinv15    = np.loadtxt('CtinvFourier_2e3steps-gxTh-t15')

#Lambdat02  = np.loadtxt('LambdaFourier_2e3steps-gxTh-t15')
#Lambdat04  = np.loadtxt('LambdaFourier_2e3steps-gxTh-t0')
#Lambdat5   = np.loadtxt('LambdaFourier_2e3steps-gxTh-t0.4')
#Lambdat15  = np.loadtxt('LambdaFourier_2e3steps-gxTh-t15')

#kinVisct02 = np.loadtxt('kinVisc_2e3steps-gxTh-t0.2')
#kinVisct04 = np.loadtxt('kinVisc_2e3steps-gxTh-t0.4')
#kinVisct5  = np.loadtxt('kinVisc_2e3steps-gxTh-t5')
#kinVisct15 = np.loadtxt('kinVisc_2e3steps-gxTh-t15')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ct0, linestyle='-', color='k')
plt.hold('on')
plt.plot(Ct02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ct04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ct2, linestyle='-', color='m')
plt.hold('on')
plt.plot(Ct5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ct10, linestyle='-', color='c')
plt.hold('on')
plt.plot(Ct15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C(t=0)$', '$C(t=0.2)$', '$C(t=0.4)$', '$C(t=2)$', '$C(t=5)$', '$C(t=10)$', '$C(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$\tilde{C}(t)$')
plt.savefig('CtFrames-56n-Fourier.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ctdev02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctdev04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctdev5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ctdev15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$\dot{C}(t=0.2)$', '$\dot{C}(t=0.4)$', '$\dot{C}(t=5)$', '$\dot{C}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$\dot{C}(t)$')
plt.savefig('CtdevFrames-56n-Fourier.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim((-4e-5, 3e-5))
ax.grid()
plt.plot(Ctdev02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctdev04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctdev5, linestyle='-', color='grey')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$\dot{C}(t=0.2)$', '$\dot{C}(t=0.4)$', '$\dot{C}(t=5)$', '$\dot{C}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$\dot{C}(t)$')
plt.savefig('CtdevFrames-56n-zoom.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid()
plt.plot(Ctinv02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctinv04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctinv5, linestyle='-', color='grey')
plt.hold('on')
plt.plot(Ctinv15, linestyle='-', color='y')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C^{-1}(t=0.2)$', '$C^{-1}(t=0.4)$', '$C^{-1}(t=5)$', '$C^{-1}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$C^{-1}(t)$')
plt.savefig('CtinvFrames-56n.eps')

plt.figure()
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim((-5e4, 1e5))
ax.grid()
plt.plot(Ctinv02, linestyle='-', color='r')
plt.hold('on')
plt.plot(Ctinv04, linestyle='-', color='g')
plt.hold('on')
plt.plot(Ctinv5, linestyle='-', color='grey')
plt.xlabel('')
plt.ylabel('')
plt.legend(['$C^{-1}(t=0.2)$', '$C^{-1}(t=0.4)$', '$C^{-1}(t=5)$', '$C^{-1}(t=15)$'], borderpad=0.1, prop={'size':10})
plt.title('$C^{-1}(t)$')
plt.savefig('CtinvFrames-56n-zoom.eps')
"""
