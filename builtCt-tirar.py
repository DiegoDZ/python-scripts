# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:59:35 2017

@author: diego
"""

import numpy as np

nNodes = 10
#Segundo índice de corr: start
h = nNodes/2
start = 0
for j in range(int(nNodes/2)):
    print h ,start
    start = start + h
    h += 1

#Segundo índice de corr: stop
k = nNodes/2 + 1
stop = 5   #Al seleccionar columnas en matrices, no pilla la columna de índice 5
for j in range(int(nNodes/2)):
    print k, stop
    stop = stop + k
    k += 1


#Unión de los dos scripts
nNodes = 10
h      = int(nNodes/2)
start  = 0
stop   = 5  #Al seleccionar columnas en matrices, no pilla la columna de índice 5
for j in range(h):
    print start, stop
    start += h
    stop  += h+1
    h += 1


#Selección de columnas de fichero correlación LAMMPS
corr = np.random.rand(5,70)  #Matriz random. Necesito saber el número de filas del output de LAMMPS
nNodes = 10
h      = int(nNodes/2)
start  = 0
stop   = 5  #Al seleccionar columnas en matrices, no pilla la columna de índice 5
for j in range(h):
    a = np.matrix((corr[:,start:stop]))
    print a.shape
    start += h
    stop  += h+1
    h += 1

#Construcción de bloques de ceros
nNodes = 10
nSteps = 5
h      = int(nNodes/2)
for j in range(h):
    print np.zeros((nSteps, h))
    h -= 1


#Unión de selección de columnas de fichero correlación LAMMPS y bloques de ceros y concatenar
import numpy as np
corr = np.random.rand(5,70) #Necesito saber el número de columnas del output de LAMMPS
nNodes = 10
nSteps = 5   #será automático
h      = int(nNodes/2)
k      = int(nNodes/2)
start  = 0
stop   = int(nNodes/2)
c0     = np.array([])
for j in range(h):
    a     = np.hstack((corr[:,start:stop],np.zeros((nSteps, k))))
    b     = np.hstack((c0,a)) if c0.size else a  #if statment acts only if c0 is empty
    c0    = b
    start += h
    stop  += h+1
    h     += 1
    k     -= 1


######################################################################
#Segunda parte. Se puede hacer por separado y luego concatenar en vertical la primera y la segunda.
######################################################################



#Segundo índice de corr: start  usa el start del anterior bucle
k = nNodes/2 + 4
for j in range(int(nNodes/2)):
    print k ,start
    start = start + k
    k -= 1

#Segundo índice de corr: stop  usa el stop del anterior bucle
k = nNodes/2 + 3
stop = stop - 1
for j in range(int(nNodes/2)):
    print k, stop
    stop = stop + k
    k -= 1

#Unión de los dos scripts
nNodes = 10
h      = int(nNodes/2)
m      = h + 4
start  = 35  #usa el start con el que acaba el anterior bucle
lastStop = 45
stop   = lastStop - 1   #usa el stop con el que acaba el anterior bucle
for j in range(h):
    print start, stop
    start += m
    stop = stop + (m - 1)
    m -= 1

#Construcción de bloques de ceros
nNodes = 10
nSteps = 5
h      = int(nNodes/2)
i = 1
for j in range(h):
    print np.zeros((nSteps, i))
    i += 1

#Construcción completa del segundo bloque
corr     = np.random.rand(5,70) #Necesito saber el número de columnas del output de LAMMPS
nNodes   = 10
nSteps   = 5   #será automático
h        = int(nNodes/2)
m        = h + 4
i        = 1
start    = 35
lastStop = 45
stop     = lastStop - 1
c0       = np.array([])
for j in range(h):
    print start, stop
    a     = np.hstack((corr[:,start:stop],np.zeros((nSteps, i))))
    b     = np.hstack((c0,a)) if c0.size else a  #if statment acts only if c0 is empty
    c0    = b
    start += m
    stop = stop + (m - 1)
    i += 1
    m -= 1


######################################################################
#Unión de los dos bloques
######################################################################
import numpy as np
corr = np.random.rand(5,70) #Necesito saber el número de columnas del output de LAMMPS
nNodes = 10
nSteps = 5   #será automático
h      = int(nNodes/2)
k      = int(nNodes/2)
start  = 0
stop   = int(nNodes/2)
c0     = np.array([])
for j in range(h):
    a     = np.hstack((corr[:,start:stop],np.zeros((nSteps, k))))
    b     = np.hstack((c0,a)) if c0.size else a  #if statment acts only if c0 is empty
    c0    = b
    start += h
    stop  += h+1
    h     += 1
    k     -= 1

h        = int(nNodes/2)
m        = h + 4
i        = 1
stop     = stop - 1
c0       = np.array([])
for j in range(h):
    d     = np.hstack((np.zeros((nSteps, i)),corr[:,start:stop]))
    e     = np.hstack((c0,d)) if c0.size else d  #if statment acts only if c0 is empty
    c0    = e
    start += m
    stop = stop + (m - 1)
    i += 1
    m -= 1

Ct = np.hstack((b,e))

#EOF

