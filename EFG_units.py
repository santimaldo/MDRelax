import numpy as np
from scipy import constants

e = constants.e
epsilon_0 = constants.epsilon_0
hbar = constants.hbar
A = 1e-10 # m
Eh,_,_ = constants.physical_constants["Hartree energy"]
a0,_,_ = constants.physical_constants["Bohr radius"]



factor = a0**2*e**3/(4*np.pi*epsilon_0*A**6*Eh)

factor = (a0/1e-10)**6

# varianza en MIS UNIDADES
# e^2/(4piepsilon0*A**6)
V2_misUnidades = 2.01e-2

V2_atomicUnits = factor*V2_misUnidades 

print("Varianza de EFG en unidades atomicas:")
print(f"<V^2> = {V2_atomicUnits:3E} a.u")
