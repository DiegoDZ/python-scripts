#----------------------------------------------------------
# THIS SCRIPT CREATES A MOVIE OF TIME CORRELATIONS FUNCTIONS
#----------------------------------------------------------
#                AUTHOR: DiegoDZ
#                DATE  : Feb2017
#                RUN   : >>> python movie-correlations.py file_name
#----------------------------------------------------------
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

#Number of frames and every how many time steps the program generates the frames
nFrames = range(1,2000,1)
#nFrames = range(1,1,1)
#----------------------------------------------------------
#LOAD FILES, DENIFE VARIABLES AND FUNCTIONS
#----------------------------------------------------------
inputFile = np.loadtxt(str(sys.argv[1]))
#Limits for corr_rhorho
#setLimitz_upper = 0.0005    # set limits of z axis
#setLimitz_lower = -0.0005  # set limits of z axis
#Limits for corr_rhogz
setLimitz_upper = 0.0005    # set limits of z axis
setLimitz_lower = -0.0005  # set limits of z axis
#Limits for corr_ee
#setLimitz_upper = 0.0005   # set limits of z axis
#setLimitz_lower = -0.0005  # set limits of z axis
#Limits for corr_gzgz
#setLimitz_upper = 0.001    # set limits of z axis
#setLimitz_lower = -0.001  # set limits of z axis

#Variables
nNodes = int(np.sqrt(len(inputFile[0])))
steps  = len(inputFile)
X      = range(0,nNodes,1)
Y      = range(0, nNodes, 1)
X, Y   = np.meshgrid(X, Y)
out    = inputFile.reshape(steps * nNodes, nNodes)

#----------------------------------------------------------
#CREATE THE FIGURE TO DO PLOTTING ON
#----------------------------------------------------------
fig = plt.figure(figsize=(14,14))
ha  = fig.gca(projection='3d')
def init():
    Z      = out[0 : nNodes, :]
    wframe = ha.plot_surface(X, Y, Z, cmap='summer', rstride=1, cstride=1, alpha=0.5, linewidth=0.3)  # Creates the frame
    return wframe

#----------------------------------------------------------
#CREATE THE MOVIE
#----------------------------------------------------------
def animate(i, ha, fig):
    ha.cla() # Clear axis
    Z = out[nNodes * i : (i + 1) * nNodes, :]
    wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=1, cstride=1, alpha=0.5, linewidth=0.3,)
    ha.set_xlabel('Nodes')
    ha.set_ylabel('Nodes')
    ha.set_zlabel('Step' + str(i))
    ha.set_zlim(setLimitz_lower, setLimitz_upper)
    return wframe

# Use FuncAnimation to create the movie using the frames.
ani = animation.FuncAnimation(fig, animate, init_func = init, frames=nFrames, fargs=(ha, fig), interval = 1)

# Save the animation as an mp4
#ani.save('animation-corr.mp4', fps=30, writer="avconv", codec="libx264")

plt.show()


