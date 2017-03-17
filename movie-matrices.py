############################################################
# This scripts creates a movie of time correlation functions
############################################################
#
# run: >> movie-Dmatrix.py
#
# Author: DiegoDZ
# Date: Feb2017
############################################################

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

################################
# Load file and define variables
################################
#D = np.loadtxt('D_rhoeg')
D = np.loadtxt('CtdevCtinv')
setLimitz_upper = 40    # set limits of z axis
setLimitz_lower = -40  # set limits of z axis
number_correlations_files = 9
number_nodes = int(np.sqrt(len(D[0]) / number_correlations_files ))
steps = len(D)
X = range(0,number_nodes * 3,1)
Y = range(0, number_nodes * 3, 1)
X, Y = np.meshgrid(X, Y)
#Select the block elements of the matrix C(t=0)
out0_00 = D[0, 0:len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_01 = D[0, len(D[0])/number_correlations_files:2 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_02 = D[0, 2*len(D[0])/number_correlations_files:3 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_10 = D[0, 3*len(D[0])/number_correlations_files:4 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_11 = D[0, 4*len(D[0])/number_correlations_files:5 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_12 = D[0, 5*len(D[0])/number_correlations_files:6 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_20 = D[0, 6*len(D[0])/number_correlations_files:7 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_21 = D[0, 7*len(D[0])/number_correlations_files:8 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
out0_22 = D[0, 8*len(D[0])/number_correlations_files:9 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
#Tr0ansform arrays into matrix
out0_00 = np.asmatrix(out0_00)
out0_01 = np.asmatrix(out0_01)
out0_02 = np.asmatrix(out0_02)
out0_10 = np.asmatrix(out0_10)
out0_11 = np.asmatrix(out0_11)
out0_12 = np.asmatrix(out0_12)
out0_20 = np.asmatrix(out0_20)
out0_21 = np.asmatrix(out0_21)
out0_22 = np.asmatrix(out0_22)
#Create the matrix Cdev(t=0)
out0 = np.bmat(([out0_00, out0_01, out0_02],[out0_10, out0_11, out0_12],[out0_20, out0_21, out0_22]))

#################################
# Create figure to do plotting on
#################################
fig = plt.figure(figsize=(14,14))
#If fig is a variable holding a figure, fig.gca() returns the axes associated with the figure. If there is a projection = ... argument to gca, the axes returned are those tagged with the indicated tag (which is commonly a string, but can be an instance of a projection class)
ha = fig.gca(projection='3d')
#Z = out[0 : number_nodes, :]
#wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=3, cstride=3, alpha=0.5, linewidth=0.3)  # Creates the frame
#fig.colorbar(wframe, shrink=0.3, aspect=6)   # Add a color bar which maps values to colors.

def init():
    Z = out0
    wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=3, cstride=3, alpha=0.5, linewidth=0.3)  # Creates the frame
    return wframe

###################
# Create the movie
###################
def animate(i, ha, fig):
    ha.cla() # Clear axis
    #Select the row (time) and the columns (correlations) and transform the arrays into matrix.
    C_00 = D[i, 0:len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_01 = D[i, len(D[0])/number_correlations_files:2 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_02 = D[i, 2*len(D[0])/number_correlations_files:3 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_10 = D[i, 3*len(D[0])/number_correlations_files:4 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_11 = D[i, 4*len(D[0])/number_correlations_files:5 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_12 = D[i, 5*len(D[0])/number_correlations_files:6 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_20 = D[i, 6*len(D[0])/number_correlations_files:7 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_21 = D[i, 7*len(D[0])/number_correlations_files:8 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_22 = D[i, 8*len(D[0])/number_correlations_files:9 * len(D[0])/number_correlations_files].reshape(number_nodes, number_nodes)
    C_00 = np.asmatrix(C_00)
    C_01 = np.asmatrix(C_01)
    C_02 = np.asmatrix(C_02)
    C_10 = np.asmatrix(C_10)
    C_11 = np.asmatrix(C_11)
    C_12 = np.asmatrix(C_12)
    C_20 = np.asmatrix(C_20)
    C_21 = np.asmatrix(C_21)
    C_22 = np.asmatrix(C_22)
    #Create the matrix C for time equal t
    C = np.bmat(([C_00, C_01, C_02],[C_10, C_11, C_12],[C_20, C_21, C_22]))
    wframe =ha.plot_surface(X, Y, C, cmap='summer', rstride=3, cstride=3, alpha=0.5, linewidth=0.3,)
    # Customize axis
    ha.set_xlabel('Nodes')
    ha.set_ylabel('Nodes')
    ha.set_zlabel('')
    ha.set_zlim(setLimitz_lower, setLimitz_upper)   #Set the limits of z axis.
    return wframe

# Use FuncAnimation to create the movie using the frames.
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = 5, fargs=(ha, fig), interval = 1)

# Save the animation as an mp4
#ani.save('animation-Dmatrix_5snapshots.mp4', fps=100, writer="avconv", codec="libx264")

plt.show()


