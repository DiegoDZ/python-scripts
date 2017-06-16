###########################################################
#THIS SCRIPT CREATES A MOVIE OF THE CORRELATION MATRIX C(t)
###########################################################
#AUTHOR: DiegoDZ
#DATE: Feb2017
#RUN: >>> movie-matrices.py
############################################################
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

#Number of frames and every how many time steps the program generates the frames
nFrames = range(0,1000,20)

###########################################################
#LOAD FILES, DENIFE VARIABLES AND FUNCTIONS
###########################################################
#D = np.loadtxt('Cmovie')
#setLimitz_upper = 0.01    # set limits of z axis
#setLimitz_lower = -0.01  # set limits of z axis
#D = np.loadtxt('C')
#setLimitz_upper = 0.01    # set limits of z axis
#setLimitz_lower = -0.01  # set limits of z axis
#D = np.loadtxt('Cpredict')
#setLimitz_upper = 0.01    # set limits of z axis
#setLimitz_lower = -0.01  # set limits of z axis
D = np.loadtxt('errorCpredict')
setLimitz_upper = 0.002    # set limits of z axis
setLimitz_lower = -0.002  # set limits of z axis
#D = np.loadtxt('Ctnorm')
#setLimitz_upper = 5    # set limits of z axis
#setLimitz_lower = -5  # set limits of z axis
#D = np.loadtxt('Ctnorminv')
#setLimitz_upper = 50    # set limits of z axis
#setLimitz_lower = -50  # set limits of z axis
#D = np.loadtxt('Ctdev')
#setLimitz_upper = 0.025    # set limits of z axis
#setLimitz_lower = -0.8  # set limits of z axis
#D = np.loadtxt('intKt')
#setLimitz_upper = 0.25    # set limits of z axis
#setLimitz_lower = -0.8  # set limits of z axis
#D = np.loadtxt('CtdevCtnorminv')
#setLimitz_upper = 1.5    # set limits of z axis
#setLimitz_lower = -1.5  # set limits of z axis
#D = np.loadtxt('Mt_rhoeg')
#setLimitz_upper = 1.5    # set limits of z axis
#setLimitz_lower = -1.5  # set limits of z axis
#D = np.loadtxt('Ft')
#setLimitz_upper = 0.025    # set limits of z axis
#setLimitz_lower = -0.8  # set limits of z axis
#D = np.loadtxt('Ut_rhoeg')
#setLimitz_upper = 5    # set limits of z axis
#setLimitz_lower = -5  # set limits of z axis
#D = np.loadtxt('MRt_system')
#setLimitz_upper = 40.0    # set limits of z axis
#setLimitz_lower = -40.0  # set limits of z axis
#D = np.loadtxt('error_MRt')
#setLimitz_upper = 1e12    # set limits of z axis
#setLimitz_lower = -1e-12  # set limits of z axis

#Variables
nBlocks = 9
sBlocks = int(np.sqrt(nBlocks))
nNodes  = int(np.sqrt(len(D[0]) / nBlocks ))
dim     = sBlocks * nNodes
steps   = len(D)
X       = range(0,dim,1)
Y       = range(0,dim,1)
X, Y    = np.meshgrid(X, Y)

#Transform vector into matrix
def reshape_vm(A):
    B = A.reshape(nBlocks,nNodes*nNodes).reshape(sBlocks,sBlocks,nNodes,nNodes).transpose(0,2,1,3).reshape(dim,dim)
    return B

###########################################################
#CREATE THE FIGURE TO DO PLOTTING ON
###########################################################
out0 = reshape_vm(D[0,:])
fig  = plt.figure(figsize=(14,14))
ha   = fig.gca(projection='3d')
def init():
    Z = out0
    wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=1, cstride=1, alpha=0.5, linewidth=0.05)
    return wframe

###########################################################
#CREATE THE MOVIE
###########################################################
def animate(t, ha, fig):
    ha.cla() # Clear axis
    C = reshape_vm(D[t,:])
    wframe =ha.plot_surface(X, Y, C, cmap='summer', rstride=1, cstride=1, alpha=0.5, linewidth=0.05,)
    ha.set_xlabel('Nodes')
    ha.set_ylabel('Nodes')
    ha.set_zlabel('Step' + str(t))
    ha.set_zlim(setLimitz_lower, setLimitz_upper)
    return wframe

#Use FuncAnimation to create the movie using the frames.
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = nFrames, fargs=(ha, fig), interval = 1)

#Save the animation as an mp4
#ani.save('animation-Dmatrix_5snapshots.mp4', fps=100, writer="avconv", codec="libx264")

plt.show()
#EOF

