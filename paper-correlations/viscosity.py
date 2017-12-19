#------------------------------------------------------------------------------
#                            viscosity.py
#------------------------------------------------------------------------------
# This script computes the viscosity from the time integral of the correlation of
# the stress tensor, the kinematic viscosity, the viscosity per bin and the
# viscosity depending on time.
# Finally, it computes the derivative of the correlation matrix, C(t), built with
# the correlation between x-momentum with itself.
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : December 2017
#                         Modified : December 2017
#                         Run      : python viscosity.py  (Run with python 2.7)
#------------------------------------------------------------------------------
import numpy as np
from scipy import linalg
import datetime

#------------------------------------------------------------------------------
# Screen questions
#------------------------------------------------------------------------------

compute_Ct        = raw_input("Do you want to increase statistics of the correlation of stress tensor? (y/n): ")
compute_eta       = raw_input("Do you want to compute the viscosity and the kinematic viscosity? (y/n): ")
compute_etaNode   = raw_input("Do you want to compute the viscosity(t) per node and the kinematic viscosity(t) per node? (y/n): ")
compute_CtdevNode = raw_input("Do you want to compute the derivative of C(t) per node (gx-Th)? (y/n): ")

#------------------------------------------------------------------------------
# Define global variables
#------------------------------------------------------------------------------

Lx,Ly,Lz   = 40.0, 40.0, 30.0
V          = Lx * Ly * Lz
T          = 2.0
nPart      = 28749
n          = nPart / V
nNodes     = 60
Vbin       = Lx * Ly * Lz / nNodes
dz         = Lz / nNodes
nBlocks    = 1
nVar       = int(np.sqrt(nBlocks))
dim        = nVar * nNodes
nSteps     = 7500
dt         = 0.004                               #lammps timestep=0.002. Saved info every 2 timesteps.
tauExp     = 0.08
rowExp     = int(tauExp/dt)
massMatrix = (4 * (np.eye(nNodes)) \
             + np.eye(nNodes, nNodes, -nNodes+1) \
             + np.eye(nNodes, nNodes, nNodes-1) + np.eye(nNodes, nNodes, -1) \
             + np.eye(nNodes, nNodes, 1)) / (6 * Vbin)
Vmatrix    = Vbin * (np.eye(nNodes))
rhoMatrix  = Vmatrix.dot(massMatrix)
laplacian  = (1 / dz**2) * (-2 * (np.eye(nNodes)) +\
          np.eye(nNodes, nNodes, -nNodes+1) + np.eye(nNodes, nNodes, nNodes-1) +
          np.eye(nNodes, nNodes, -1) + np.eye(nNodes, nNodes, 1))

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

#Take advantage of the periodic boundary conditions in the creation of C(t) matrix.
# !!! Only for system 'fluid' with periodic boundary conditions.
def pbc(C):
    Cstat = np.zeros((nNodes,nNodes))
    for i in range(nNodes):
        for j in range(nNodes):
            for k in range(nNodes):
                Cstat[i,j] += C[(i+k)%nNodes,(j+k)%nNodes]
    return Cstat/nNodes

#------------------------------------------------------------------------------
# Start computation
#------------------------------------------------------------------------------

# C(t) matrix from correlation file corr_stress-f-avg (only for taking advantage of periodic boundary condictions).
if compute_Ct == 'y':
    Ct       = np.zeros((nSteps, nBlocks * nNodes ** 2))
    c_stress = np.loadtxt('corr_stress-f-avg')
    for i in range(nSteps):
        print datetime.datetime.now(), 'Increasing statistics of sigma-xz. Step', str(i)
        Ct[i,:] = reshape_mv(((pbc(reshape_vm(c_stress[i,:])) + pbc(reshape_vm(c_stress[i,:]).T)) / 2))
    #Save result
    np.savetxt('Ct-Sxz', Ct)
    print datetime.datetime.now(), 'C(t) with sigma-xz computed.'

# Compute viscosity and kinematic viscosity
if compute_eta == 'y':
    #Compute viscosity for tauExp or all support.
    print datetime.datetime.now(), 'Computing viscosity and kinematic viscosity...'
    Ct             = np.loadtxt('Ct-Sxz')
    viscosity      = np.trapz(Ct[0:rowExp,:], dx=dt, axis=0) / T
    viscosity_t    = np.trapz(Ct, dx=dt, axis=0) / T                                        # all time
    kinViscosity   = reshape_vm(viscosity).dot(Vmatrix).dot(linalg.pinv(rhoMatrix))
    kinViscosity_t = reshape_vm(viscosity_t).dot(Vmatrix).dot(linalg.pinv(rhoMatrix))       # all time
    #Save results
    np.savetxt('viscosity-GK',      reshape_vm(viscosity))
    np.savetxt('viscosity_t-GK',    reshape_vm(viscosity_t))
    np.savetxt('kinViscosity-GK',   kinViscosity)
    np.savetxt('kinViscosity_t-GK', kinViscosity_t)
    print datetime.datetime.now(), 'Viscosity and kinematic viscosity matrix computed.'

# Compute viscosity and kinematic viscosity per node. Also viscosity(t).
if compute_etaNode == 'y':
    print datetime.datetime.now(), 'Computing viscosity and kinematic viscosity per node...'
    Ct          = np.loadtxt('Ct-Sxz')
    visc        = np.zeros((nSteps, nNodes**2))
    lapVisc     = np.zeros((nSteps, nNodes**2))
    nodeVisc    = np.zeros((nSteps, nNodes))
    nodeLapVisc = np.zeros((nSteps, nNodes))
    nodeKinVisc = np.zeros((nSteps, nNodes))
    for i in range(nSteps):
        visc[i,:]        = np.trapz(Ct[0:i+1,:], dx=dt, axis=0)
        lapVisc[i,:]     = reshape_mv(laplacian.dot(reshape_vm(visc[i,:])))
        nodeVisc[i,:]    = np.sum(reshape_vm(visc[i,:]), axis=1)
        nodeLapVisc[i,:] = np.sum(reshape_vm(lapVisc[i,:]), axis=1)
        nodeKinVisc[i,:] = np.sum(reshape_vm(visc[i,:]).dot(Vmatrix).dot(linalg.pinv(rhoMatrix)), axis=1)
    nodeVisc    /= nNodes
    nodeKinVisc /= nNodes
    visc    /= T
    lapVisc /= T
    #Save results
    np.savetxt('viscosityt', visc)
    np.savetxt('LapViscosityt', lapVisc)
    np.savetxt('nodeViscosity', nodeVisc)
    np.savetxt('nodeLapViscosity', nodeLapVisc)
    np.savetxt('nodeKinViscosity', nodeKinVisc)
    print datetime.datetime.now(), 'Viscosity and kinematic viscosity per node computed.'

# Compute the derivative of C(t)
if compute_CtdevNode == 'y':
    dt             = 0.002          #!!!the x-momentum was calculated using simulations with dt=0.0002 (saved info every 10 timesteps)
    nSteps         = 15000
    Ct_gxTh        = np.loadtxt('Ct-gxTh')
    Ctdev          = np.zeros((nSteps, nNodes**2))
    CtNode         = np.zeros((nSteps, nNodes))
    CtdevNode      = np.zeros((nSteps, nNodes))
    Ctdev0         = (reshape_vm(Ct_gxTh[1,:]) - reshape_vm(Ct_gxTh[1,:]).T) / (2*dt)
    Ctdev[0,:]     = reshape_mv(Ctdev0)
    CtdevNode[0,:] = np.sum(Ctdev0, axis=1)
    for i in range(1, nSteps-1):
        Cforward       = reshape_vm(Ct_gxTh[i+1,:])
        Cbackward      = reshape_vm(Ct_gxTh[i-1,:])
        Ctdev[i,:]     = reshape_mv((Cforward - Cbackward) / (2*dt))
        CtdevMatrix    = (Cforward - Cbackward) / (2*dt)
        CtdevNode[i,:] = np.sum(CtdevMatrix, axis=1)
    CtdevNode /= nNodes
    for j in range(nSteps):
        CtNode[j,:]    = np.sum(reshape_vm(Ct_gxTh[j,:]), axis=1)
    #Save results
    CtNode /= nNodes
    np.savetxt('Ctdev-gxTh', Ctdev)
    np.savetxt('CtNode-gxTh', CtNode)
    np.savetxt('CtdevNode-gxTh', CtdevNode)

#EOF
