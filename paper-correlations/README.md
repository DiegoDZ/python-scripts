#python-scripts

Python scripts used to obtain the results of __On the non-locality of hydrodynamics...__

__computeLambda.py__ computes the matrix of correlations C(t) from correlation files. In this paper we only use the theory with the x-momentum of the fluid as relevant variable. Also the matrix Lambda is computed.
__computeLambdaFourier.py__ computes the matrix of correlations C(t) in the Fourier space taking advantage of the periodic boundary conditions. The predictions of the model and the viscosity are computed as well.
__viscosity.py__ computes the shear and the kinematic viscosity of the fluid. In this script we calculate the derivative of the matrix of correlations computed in __computeLambda.py__ in order to prove that its decay is proportional to the decay of the viscosuty.  


