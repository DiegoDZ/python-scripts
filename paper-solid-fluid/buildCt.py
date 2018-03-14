#------------------------------------------------------------------------------
#                            buildCt.py
#------------------------------------------------------------------------------
# This script builds the matrix of correlations C(t) only with a bunch of
# correlations computing with LAMMPS.
#------------------------------------------------------------------------------
#                         Author   : @DiegoDZ
#                         Date     : January  2018
#                         Modified : February 2018
#                         Run      : python buildCt.py  (Run with python 2.7)
#------------------------------------------------------------------------------

import numpy as np


#--------------------------------------Screen question-----------------------------
gxgx     = raw_input("Do you want to build C(t)? (y/n):")
SxzFx    = raw_input("Do you want to build <Sxz(t)Sxz>, <Sxz(t)Fx>, <Fx(t)Sxz>, <Fx(t)Fx>? (y/n):")


#----------------------------------
#Inputs
#----------------------------------
nSteps, nCols = 7500, 400
#nNodesSys     = 66
nNodes        = 61
nBlocks       = 1                                         #Do not confuse with the number of blocks we will use to build the matrix
                                                          #of correlations C(t). This number of blocks refers to the number of different
                                                          #correlation files (i.e. <gxgx(t)>, <gxrho(t)> ...).
                                                          #In this case the number of blocks is only one because the matrix of
                                                          #correlations will be <gxgx(t)>.
sBlocks       = int(np.sqrt(nBlocks))
dim           = sBlocks * nNodes
#We define the number of nodes for the blocks calculated with LAMMPS. Note the different between the number of nodes of C(t) and the number of nodes of block1 and block2.
nNodesBlock   = int(np.sqrt(nCols))
dimBlock      = sBlocks * nNodesBlock

#dimSys = nNodesSys
#----------------------------------
#Subrutines
#----------------------------------
#We will distinguish between the reshapes for the block1 and block2 and the reshapes for the matrix of correlations C(t).

#Change format: vector-> matrix  (For block1 and block2)
def reshapeBlock_vm(A):
    B = A.reshape(nBlocks,nNodesBlock*nNodesBlock).reshape(sBlocks,sBlocks,nNodesBlock,nNodesBlock).transpose(0,2,1,3).reshape(dimBlock,dimBlock)
    return B

#Change format: matrix-> vector  (For blocks)
def reshapeBlock_mv(A):
    B = A.reshape(sBlocks,nNodesBlock,sBlocks,nNodesBlock).swapaxes(1,2).ravel()
    return B

#Change format: vector-> matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

#Change format: matrix-> vector
def reshape_mv(A):
    B = A.reshape(sBlocks,nNodes,sBlocks,nNodes).swapaxes(1,2).ravel()
    return B

#def reshape_mv_new(A):
#    B = A.reshape(sBlocks,nNodesSys,sBlocks,nNodesSys).swapaxes(1,2).ravel()
#    return B
#def reshape_vm_new(A):
#    B = A.reshape(nBlocks,nNodesSys*nNodesSys).reshape(sBlocks,sBlocks,nNodesSys,nNodesSys).transpose(0,2,1,3).reshape(dimSys,dimSys)
#    return B

#Average blocks of correlations. Note that this subrutine is for blocks of correlations in which a variable defined in bins is involved. This is the reason why it does not take the last row and column.
def avgBlockSxzSxz(B1,B2):
    A = 0.5 * (reshapeBlock_vm(B1)[0:-1, 0:-1] + np.rot90(reshapeBlock_vm(B2)[0:-1, 0:-1],2))
    reshapeBlock_vm(B1)[0:-1, 0:-1] = A
    reshapeBlock_vm(B2)[0:-1, 0:-1] = np.rot90(A,2)
    B1avg = reshapeBlock_mv(B1)
    B2avg = reshapeBlock_mv(B2)
    return B1avg, B2avg
def avgBlockSxzFx(B1,B2):
    A = 0.5 * (reshapeBlock_vm(B1)[0:-1,:] + np.rot90(reshapeBlock_vm(B1),4)[1:,:])
    B = 0.5 * (reshapeBlock_vm(B2)[0:-1,:] + np.rot90(reshapeBlock_vm(B2),4)[1:,:])
    reshapeBlock_vm(B1)[0:-1, :] = A
    reshapeBlock_vm(B2)[0:-1, :] = B
    #reshapeBlock_vm(B2)[0:-1, :] = np.rot90(A,2)
    B1avg = reshapeBlock_mv(B1)
    B2avg = reshapeBlock_mv(B2)
    return B1avg, B2avg
def avgBlockFxSxz(B1,B2):
    A = 0.5 * (reshapeBlock_vm(B1)[:, 0:-1] + np.rot90(reshapeBlock_vm(B1),4)[:,1:])
    B = 0.5 * (reshapeBlock_vm(B2)[:, 0:-1] + np.rot90(reshapeBlock_vm(B2),4)[:,1:])
    reshapeBlock_vm(B1)[:,0:-1] = A
    reshapeBlock_vm(B2)[:,0:-1] = B
    #reshapeBlock_vm(B2)[:,0:-1] = np.rot90(A,2)
    B1avg = reshapeBlock_mv(B1)
    B2avg = reshapeBlock_mv(B2)
    return B1avg, B2avg

#Obtain the anti-diagonal (and the next one) of a block. Before that the subrutine use all the sub-anti-diagonals to increase the stat. of the main anti-diagonal.
def avgDiag(B):
    Davg = np.zeros(nNodesBlock)
    k = np.zeros(nNodesBlock)
    j = 1
    for i in np.arange(0, nNodesBlock, 2):
        D = np.diag(np.rot90(B,1), k=i)
        if i!=0:
            DZ = np.concatenate([np.zeros(i-j), D, np.zeros(i-j)])
            j += 1
        else:
            DZ = D
        for index in np.ndindex(DZ.shape):
            if DZ[index]!=0:
                Davg[index] += DZ[index]
                k[index] += 1
            else:
                Davg[index] = Davg[index]
    Davg /= k

    D1avg = np.zeros(nNodesBlock-1)
    k    = np.zeros(nNodesBlock-1)
    j = 1
    for i in np.arange(1, nNodesBlock, 2):
        D = np.diag(np.rot90(B,1), k=i)
        DZ = np.concatenate([np.zeros(i-j), D, np.zeros(i-j)])
        j += 1
        for index in np.ndindex(DZ.shape):
            if DZ[index]!=0:
                D1avg[index] += DZ[index]
                k[index] += 1
            else:
                D1avg[index] = D1avg[index]
    D1avg /= k
    return Davg, D1avg

#Build the matrix of correlations when the diagonal does have to be built (<gx(t)gx> and <Sxz(t)Sxz>).
def buildC(C, Davg, D1avg):
    for (k, k1) in zip(range(nNodesBlock), range(nNodesBlock-1)):
        for i in np.arange(nNodesBlock-1, nNodes, 1):
            C[i-k,(i+1-k)-nNodesBlock+2*k]  = Davg[k]
            C[i-k,(i-k1+2)-nNodesBlock+2*k] = D1avg[k1]
    return C

#Build the matrix of correlations when the diagonal does not have to be built (<Sxz(t)Fx>, <Fx(t)Sxz> and <Fx(t)Fx>).
def buildC2(corrB1,corrB2):
    Z = np.zeros((nNodes, nNodes))
    B1 = reshapeBlock_vm(corrB1)
    B2 = reshapeBlock_vm(corrB2)
    Z[0:nNodesBlock,0:nNodesBlock]                          = B1
    Z[nNodes-nNodesBlock:nNodes, nNodes-nNodesBlock:nNodes] = B2
    C = reshape_mv(Z)
    return C

#----------------------------------------------------------MATRIX C(t)---------------------------------------------------------

if gxgx=='y':

    block1 = np.loadtxt('corr-gxgx-sf-block1-AVG.dat')
    block2 = np.loadtxt('corr-gxgx-sf-block2-AVG.dat')

    corr_gxgx_B1_avg   = np.zeros((nSteps, nNodesBlock**2))
    corr_gxgx_B2_avg   = np.zeros((nSteps, nNodesBlock**2))

    for t in range(nSteps):
        B1 = reshapeBlock_mv(0.5 * (reshapeBlock_vm(block1[t,:]) + reshapeBlock_vm(block1[t,:]).T))
        B2 = reshapeBlock_mv(0.5 * (reshapeBlock_vm(block2[t,:]) + reshapeBlock_vm(block2[t,:]).T))
        A  = 0.5 * (reshapeBlock_vm(B1) + np.rot90(reshapeBlock_vm(B2),2))
        corr_gxgx_B1_avg[t,:] = reshapeBlock_mv(A)
        corr_gxgx_B2_avg[t,:] = reshapeBlock_mv(np.rot90(A,2))
    np.savetxt('C-sf-B1-t0.dat'   , reshapeBlock_vm(corr_gxgx_B1_avg[0,:]))
    np.savetxt('C-sf-B1-t0.2.dat' , reshapeBlock_vm(corr_gxgx_B1_avg[50,:]))
    np.savetxt('C-sf-B2-t0.dat'   , reshapeBlock_vm(corr_gxgx_B2_avg[0,:]))
    np.savetxt('C-sf-B2-t0.2.dat' , reshapeBlock_vm(corr_gxgx_B2_avg[50,:]))

    C = np.zeros((nSteps,nNodes**2))
    for t in range(nSteps):
        print 'Building <gx(t)gx>. Step '+ str(t+1)
        Z  = np.zeros((nNodes, nNodes))
        B1 = reshapeBlock_vm(corr_gxgx_B1_avg[t,:])
        B2 = reshapeBlock_vm(corr_gxgx_B2_avg[t,:])
        Z[0:nNodesBlock,0:nNodesBlock]                          = B1
        Z[nNodes-nNodesBlock:nNodes, nNodes-nNodesBlock:nNodes] = B2
        Davg, D1avg = avgDiag(B1)
        C[t,:] = reshape_mv(buildC(Z,Davg,D1avg))
    np.savetxt('Ct-sf.dat'          , C)
    np.savetxt('Ct-sf-500steps.dat' , C[0:500,:])
    np.savetxt('C-sf-t0.dat'  , reshape_vm(C[0,:]))
    np.savetxt('C-sf-t0.2.dat', reshape_vm(C[50,:]))

#--------------------------------------------------<Sxz(t)Fx>, <Fx(t)Sxz> and <Fx(t)Fx>---------------------------------------

if SxzFx == 'y':
    block1 = np.loadtxt('corr-SxzFx-sf-block1-AVG.dat')
    block2 = np.loadtxt('corr-SxzFx-sf-block2-AVG.dat')

    nRows, nCols = np.shape(block1)
    nNodesBlock  = int(np.sqrt(nCols/4))

#---------------Obtain blocks of correlations---------------

    corr_SxzSxz_B1 = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzSxz_B2 = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzFx_B1  = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzFx_B2  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxSxz_B1  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxSxz_B2  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxFx_B1   = np.zeros((nSteps, nNodesBlock**2))
    corr_FxFx_B2   = np.zeros((nSteps, nNodesBlock**2))

    for t in range(nSteps):
        print 'Creating blocks. Step '+ str(t+1)
        B1 = block1[t,:].reshape(2*nNodesBlock, 2*nNodesBlock)
        B2 = block2[t,:].reshape(2*nNodesBlock, 2*nNodesBlock)

        corr_SxzSxz_B1[t,:] = reshapeBlock_mv(B1[0*nNodesBlock:1*nNodesBlock, 0*nNodesBlock:1*nNodesBlock])
        corr_SxzSxz_B2[t,:] = reshapeBlock_mv(B2[0*nNodesBlock:1*nNodesBlock, 0*nNodesBlock:1*nNodesBlock])
        corr_SxzFx_B1[t,:]  = reshapeBlock_mv(B1[0*nNodesBlock:1*nNodesBlock, 1*nNodesBlock:2*nNodesBlock])
        corr_SxzFx_B2[t,:]  = reshapeBlock_mv(B2[0*nNodesBlock:1*nNodesBlock, 1*nNodesBlock:2*nNodesBlock])
        corr_FxSxz_B1[t,:]  = reshapeBlock_mv(B1[1*nNodesBlock:2*nNodesBlock, 0*nNodesBlock:1*nNodesBlock])
        corr_FxSxz_B2[t,:]  = reshapeBlock_mv(B2[1*nNodesBlock:2*nNodesBlock, 0*nNodesBlock:1*nNodesBlock])
        corr_FxFx_B1[t,:]   = reshapeBlock_mv(B1[1*nNodesBlock:2*nNodesBlock, 1*nNodesBlock:2*nNodesBlock])
        corr_FxFx_B2[t,:]   = reshapeBlock_mv(B2[1*nNodesBlock:2*nNodesBlock, 1*nNodesBlock:2*nNodesBlock])
    np.savetxt('corr-SxzSxz-sf-B1-t0.dat', reshapeBlock_vm(corr_SxzSxz_B1[0,:]))
    np.savetxt('corr-SxzSxz-sf-B2-t0.dat', reshapeBlock_vm(corr_SxzSxz_B2[0,:]))
    np.savetxt('corr-SxzFx-sf-B1-t0.dat' , reshapeBlock_vm(corr_SxzFx_B1[0,:]))
    np.savetxt('corr-SxzFx-sf-B2-t0.dat' , reshapeBlock_vm(corr_SxzFx_B2[0,:]))
    np.savetxt('corr-FxSxz-sf-B1-t0.dat' , reshapeBlock_vm(corr_FxSxz_B1[0,:]))
    np.savetxt('corr-FxSxz-sf-B2-t0.dat' , reshapeBlock_vm(corr_FxSxz_B2[0,:]))
    np.savetxt('corr-FxFx-sf-B1-t0.dat'  , reshapeBlock_vm(corr_FxFx_B1[0,:]))
    np.savetxt('corr-FxFx-sf-B2-t0.dat'  , reshapeBlock_vm(corr_FxFx_B2[0,:]))
    np.savetxt('corr-SxzSxz-sf-B1-t0.2.dat', reshapeBlock_vm(corr_SxzSxz_B1[50,:]))
    np.savetxt('corr-SxzSxz-sf-B2-t0.2.dat', reshapeBlock_vm(corr_SxzSxz_B2[50,:]))
    np.savetxt('corr-SxzFx-sf-B1-t0.2.dat' , reshapeBlock_vm(corr_SxzFx_B1[50,:]))
    np.savetxt('corr-SxzFx-sf-B2-t0.2.dat' , reshapeBlock_vm(corr_SxzFx_B2[50,:]))
    np.savetxt('corr-FxSxz-sf-B1-t0.2.dat' , reshapeBlock_vm(corr_FxSxz_B1[50,:]))
    np.savetxt('corr-FxSxz-sf-B2-t0.2.dat' , reshapeBlock_vm(corr_FxSxz_B2[50,:]))
    np.savetxt('corr-FxFx-sf-B1-t0.2.dat'  , reshapeBlock_vm(corr_FxFx_B1[50,:]))
    np.savetxt('corr-FxFx-sf-B2-t0.2.dat'  , reshapeBlock_vm(corr_FxFx_B2[50,:]))

#---------------Average blocks---------------

    corr_SxzSxz_B1_avg = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzSxz_B2_avg = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzFx_B1_avg  = np.zeros((nSteps, nNodesBlock**2))
    corr_SxzFx_B2_avg  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxSxz_B1_avg  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxSxz_B2_avg  = np.zeros((nSteps, nNodesBlock**2))
    corr_FxFx_B1_avg   = np.zeros((nSteps, nNodesBlock**2))
    corr_FxFx_B2_avg   = np.zeros((nSteps, nNodesBlock**2))

    for t in range(nSteps):
        print 'Averaging blocks. Step '+ str(t+1)
        #<Sxz(t)Sxz>
        B1avg, B2avg = avgBlockSxzSxz(corr_SxzSxz_B1[t,:], corr_SxzSxz_B2[t,:])
        B1avg = reshapeBlock_mv(0.5*(reshapeBlock_vm(B1avg)+ reshapeBlock_vm(B1avg).T))   #symmetric
        B2avg = reshapeBlock_mv(0.5*(reshapeBlock_vm(B2avg)+ reshapeBlock_vm(B2avg).T))   #symmetric
        corr_SxzSxz_B1_avg[t,:] = B1avg
        corr_SxzSxz_B2_avg[t,:] = B2avg
        #<Sxz(t)Fx>
        #B1avg, B2avg = avgBlockSxzFx(corr_SxzFx_B1[t,:], corr_SxzFx_B2[t,:])
        #corr_SxzFx_B1_avg[t,:] = B1avg
        #corr_SxzFx_B2_avg[t,:] = B2avg
        corr_SxzFx_B1_avg[t,:] = corr_SxzFx_B1[t,:]
        corr_SxzFx_B2_avg[t,:] = corr_SxzFx_B2[t,:]
        #<Fx(t)Sxz>
        #B1avg, B2avg = avgBlockFxSxz(corr_FxSxz_B1[t,:], corr_FxSxz_B2[t,:])
        #corr_FxSxz_B1_avg[t,:] = B1avg
        #corr_FxSxz_B2_avg[t,:] = B2avg
        corr_FxSxz_B1_avg[t,:] = corr_FxSxz_B1[t,:]
        corr_FxSxz_B2_avg[t,:] = corr_FxSxz_B2[t,:]
        #<Fx(t)Fxz>    *Note the difference between the following piece of code and the later one. This is because the force is defined in nodes and the stress tensor in bins -> there is an extra zero
        #A = 0.5 * (reshapeBlock_vm(corr_FxFx_B1[t,:]) + np.rot90(reshapeBlock_vm(corr_FxFx_B2[t,:]),2))
        #corr_FxFx_B1_avg[t,:] = reshapeBlock_mv(A)
        #corr_FxFx_B2_avg[t,:] = reshapeBlock_mv(np.rot90(A,2))
        corr_FxFx_B1_avg[t,:] = corr_FxFx_B1[t,:]
        corr_FxFx_B2_avg[t,:] = corr_FxFx_B2[t,:]

    np.savetxt('corr-SxzSxz-sf-B1-avg-t0.dat'  , reshapeBlock_vm(corr_SxzSxz_B1_avg[0,:]))
    np.savetxt('corr-SxzSxz-sf-B2-avg-t0.dat'  , reshapeBlock_vm(corr_SxzSxz_B2_avg[0,:]))
    np.savetxt('corr-SxzFx-sf-B1-avg-t0.dat'   , reshapeBlock_vm(corr_SxzFx_B1_avg[0,:]))
    np.savetxt('corr-SxzFx-sf-B2-avg-t0.dat'   , reshapeBlock_vm(corr_SxzFx_B2_avg[0,:]))
    np.savetxt('corr-FxSxz-sf-B1-avg-t0.dat'   , reshapeBlock_vm(corr_FxSxz_B1_avg[0,:]))
    np.savetxt('corr-FxSxz-sf-B2-avg-t0.dat'   , reshapeBlock_vm(corr_FxSxz_B2_avg[0,:]))
    np.savetxt('corr-FxFx-sf-B1-avg-t0.dat'    , reshapeBlock_vm(corr_FxFx_B1_avg[0,:]))
    np.savetxt('corr-FxFx-sf-B2-avg-t0.dat'    , reshapeBlock_vm(corr_FxFx_B2_avg[0,:]))
    np.savetxt('corr-SxzSxz-sf-B1-avg-t0.2.dat', reshapeBlock_vm(corr_SxzSxz_B1_avg[50,:]))
    np.savetxt('corr-SxzSxz-sf-B2-avg-t0.2.dat', reshapeBlock_vm(corr_SxzSxz_B2_avg[50,:]))
    np.savetxt('corr-SxzFx-sf-B1-avg-t0.2.dat' , reshapeBlock_vm(corr_SxzFx_B1_avg[50,:]))
    np.savetxt('corr-SxzFx-sf-B2-avg-t0.2.dat' , reshapeBlock_vm(corr_SxzFx_B2_avg[50,:]))
    np.savetxt('corr-FxSxz-sf-B1-avg-t0.2.dat' , reshapeBlock_vm(corr_FxSxz_B1_avg[50,:]))
    np.savetxt('corr-FxSxz-sf-B2-avg-t0.2.dat' , reshapeBlock_vm(corr_FxSxz_B2_avg[50,:]))
    np.savetxt('corr-FxFx-sf-B1-avg-t0.2.dat'  , reshapeBlock_vm(corr_FxFx_B1_avg[50,:]))
    np.savetxt('corr-FxFx-sf-B2-avg-t0.2.dat'  , reshapeBlock_vm(corr_FxFx_B2_avg[50,:]))

#---------------Build matrices of correlations---------------

    C = np.zeros((nSteps,nNodes**2))
    #CSys = np.zeros((nSteps,nNodesSys**2))
    for t in range(nSteps):
        print 'Building <Sxz(t)Sxz>. Step '+ str(t+1)
        Z  = np.zeros((nNodes, nNodes))
        B1 = reshapeBlock_vm(corr_SxzSxz_B1_avg[t,:])
        B2 = reshapeBlock_vm(corr_SxzSxz_B2_avg[t,:])
        Z[0:nNodesBlock,0:nNodesBlock]                          = B1
        Z[nNodes-nNodesBlock:nNodes, nNodes-nNodesBlock:nNodes] = B2
        #Z[nNodes-nNodesBlock:nNodes, nNodes-nNodesBlock:nNodes] = np.rot90(B1,2)
        Davg, D1avg = avgDiag(B1)
        C[t,:] = reshape_mv(buildC(Z,Davg,D1avg))
        
        #Nuevo
    #    Zprev = np.zeros((nNodesSys, nNodesSys))
    #    Cprev = buildC(Z,Davg,D1avg)
    #    Zprev[4:65,4:65] = Cprev
    #    CSys[t,:] = reshape_mv_new(Zprev)
    np.savetxt('corr-SxzSxz-sf.dat', C)
    np.savetxt('corr-SxzSxz-sf-500steps.dat', C[0:500])
    np.savetxt('corr-SxzSxz-sf-t0.dat'  , reshape_vm(C[0,:]))
    np.savetxt('corr-SxzSxz-sf-t0.2.dat', reshape_vm(C[50,:]))
    #np.savetxt('corr-SxzSxz-sf-t0.2.dat-new', reshape_vm_new(CSys[50,:]))

    C1 = np.zeros((nSteps,nNodes**2))
    C2 = np.zeros((nSteps,nNodes**2))
    C3 = np.zeros((nSteps,nNodes**2))
    #C1Sys = np.zeros((nSteps,nNodesSys**2))
    #C2Sys = np.zeros((nSteps,nNodesSys**2))
    #C3Sys = np.zeros((nSteps,nNodesSys**2))
    for t in range(nSteps):
        print 'Building <Sxz('+str(t+1)+')Fx>, <Fx('+str(t+1)+')Sxz> and <Fx('+str(t+1)+')Fx>. Step '+ str(t+1)
        #C1[t,:] = buildC2(corr_SxzFx_B1_avg[t,:], reshapeBlock_mv(np.rot90(reshapeBlock_vm(corr_SxzFx_B1_avg[t,:]),2)))
        #C2[t,:] = buildC2(corr_FxSxz_B1_avg[t,:], reshapeBlock_mv(np.rot90(reshapeBlock_vm(corr_FxSxz_B1_avg[t,:]),2)))
        C1[t,:] = buildC2(corr_SxzFx_B1_avg[t,:], corr_SxzFx_B2_avg[t,:])
        C2[t,:] = buildC2(corr_FxSxz_B1_avg[t,:], corr_FxSxz_B2_avg[t,:])
        C3[t,:] = buildC2(corr_FxFx_B1_avg[t,:] , corr_FxFx_B2_avg[t,:])

    #    Z1prev = np.zeros((nNodesSys, nNodesSys))
    #    Z2prev = np.zeros((nNodesSys, nNodesSys))
    #    Z3prev = np.zeros((nNodesSys, nNodesSys))
    #    Z1prev[4:65,3:64] = reshape_vm(C1[t,:])
    #    Z2prev[4:65,3:64] = reshape_vm(C2[t,:])
    #    Z3prev[3:64,4:65] = reshape_vm(C3[t,:])
    #    C1Sys[t,:] = reshape_mv_new(Z1prev)
    #    C2Sys[t,:] = reshape_mv_new(Z2prev)
    #    C3Sys[t,:] = reshape_mv_new(Z3prev)

    #np.savetxt('corr-SxzFx-sf-t0.2.dat-new', reshape_vm_new(C1Sys[50,:]))
    #np.savetxt('corr-FxSxz-sf-t0.2.dat-new', reshape_vm_new(C2Sys[50,:]))
    #np.savetxt('corr-FxFx-sf-t0.2.dat-new', reshape_vm_new(C3Sys[50,:]))


    np.savetxt('corr-SxzFx-sf.dat', C1)
    np.savetxt('corr-FxSxz-sf.dat', C2)
    np.savetxt('corr-FxFx-sf.dat' , C3)
    np.savetxt('corr-SxzFx-sf-500steps.dat', C1[0:500,:])
    np.savetxt('corr-FxSxz-sf-500steps.dat', C2[0:500,:])
    np.savetxt('corr-FxFx-sf-500steps.dat' , C3[0:500,:])
    np.savetxt('corr-SxzFx-sf-t0.dat'  , reshape_vm(C1[0,:]))
    np.savetxt('corr-FxSxz-sf-t0.dat'  , reshape_vm(C2[0,:]))
    np.savetxt('corr-FxFx-sf-t0.dat'   , reshape_vm(C3[0,:]))
    np.savetxt('corr-SxzFx-sf-t0.2.dat', reshape_vm(C1[50,:]))
    np.savetxt('corr-FxSxz-sf-t0.2.dat', reshape_vm(C2[50,:]))
    np.savetxt('corr-FxFx-sf-t0.2.dat' , reshape_vm(C3[50,:]))

#EOF

