############################################################
# This scripts creates a movie of time correlation functions
############################################################
#
# run: >> movie-correlations.py file_name
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
inputFile = np.loadtxt(str(sys.argv[1]))
setLimitz_upper = 0.001    # set limits of z axis
setLimitz_lower = -0.0002  # set limits of z axis
number_nodes = int(np.sqrt(len(inputFile[0])))
steps = len(inputFile)
X = range(0,number_nodes,1)
Y = range(0, number_nodes, 1)
X, Y = np.meshgrid(X, Y)
out = inputFile.reshape(steps * number_nodes, number_nodes)   #reshape the file

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
    Z = out[0 : number_nodes, :]
    wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=3, cstride=3, alpha=0.5, linewidth=0.3)  # Creates the frame
    return wframe

###################
# Create the movie
###################
def animate(i, ha, fig):
    ha.cla() # Clear axis
    Z = out[number_nodes * i : (i + 1) * number_nodes, :]
    wframe =ha.plot_surface(X, Y, Z, cmap='summer', rstride=3, cstride=3, alpha=0.5, linewidth=0.3,)
    # Customize axis
    ha.set_xlabel('Nodes')
    ha.set_ylabel('Nodes')
    ha.set_zlabel('')
    ha.set_zlim(setLimitz_lower, setLimitz_upper)   #Set the limits of z axis.
    return wframe

# Use FuncAnimation to create the movie using the frames.
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = 500, fargs=(ha, fig), interval = 1)

# Save the animation as an mp4
#ani.save('animation-corr.mp4', fps=30, writer="avconv", codec="libx264")

plt.show()


