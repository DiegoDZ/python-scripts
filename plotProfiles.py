# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:32:24 2017

@author: diego
"""
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

#For meso variables
X = range(0,100,1)
Y = range(0, 100, 1)
data = np.loadtxt('rho_evolution_rhoeTheory_100')
print data.shape
#data = data[0:100,:]
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(X, Y, data,cmap='summer', rstride=1, cstride=1, alpha=0.2, linewidth=0.1)
ax.set_xlabel('Nodes')
ax.set_ylabel('Time step')
#ax.set_zlabel('Fluid momentum') 
ax.text2D(0.01, 0.9, "Fluid density MODEL", transform=ax.transAxes)        
fig.savefig('fluiddensityMODEL.png', papertype = 'a0', format='png')     
plt.show()

#For macro variables
data = np.loadtxt('macroEnergy_wall')
data = data[0:1000]
x = range(0,1000,1)
plt.plot(x,data, 'g')
plt.grid(True, which='both')
plt.xlabel('Nodes')
plt.ylabel('Solid energy')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('solidenergy.png', papertype = 'a0', format='png')  
plt.show()
