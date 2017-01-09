#python-scripts

Important python scritps used in my doctoral thesis. 

**General scripts**
-------------------

__covariance_mm.py__ Computes the covariance between two input matrix. The output is a matrix.

__covariance_mv.py__ Computes the covariance between a matrix and a vector. The output is a vector.

__covariance_vm.py__ Computes the covariance between a vector and a matrix. The output is a vector.

__covariance_vv.py__ Computes the covariance between two input vectors. The output is a vector.

__projected-current.py__ Computes the projected current of some relevant variables (internal energy of the fluid, heat fluxes and time derivative of the internal energy of the fluid)

__Hmatrix.py__ Computes the matrix of covariances H. 

__Mmatrix.py__ Computes the matrix M from K,L,N and S, and from the correlations between iLepsilon with iLepsilon.


**Specific scripts used in correlation.job**
--------------------------------------------

__average-correlations.py__  Averages the correlation functions.

__average-covariances.py__ Averages the covariances. 

__average_profiles.py__ Averages the output profiles. 

__nodes-selection.py__ Selects the fluid nodes. 

**Specific scripts used in covariances.job and configurations.job**
-----------------------------------------------------------------
__average-results.py__ Averages the covariances. 

