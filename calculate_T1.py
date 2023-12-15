#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:05:46 2023

@author: santi

Calculo T1
"""
import numpy as np

e = 1.60217663 * 1e-19  # Coulomb
hbar = 1.054571817 * 1e-34  # joule seconds
ke = 8.9875517923 * 1e9  # Vm/C, Coulomb constant

Q = -4.01 * (1e-15)**2  # m**2
I = 1.5  # spin 3/2


gamma = 0.17  # Sternhemmer factor


# DME
efg_variance = 0.0206515220481938* ke**2 * e**2 / (1e-10)**6 # (V/m)^2
### integral, plateau a ojo:
# tau_c = 5 * 1e-12 # s

### Metodo triexp:
As = np.array([0.34845, 0.13996, 0.49357])
taus = np.array([0.03717, 1.01674, 11.13488])*1e-12
tau_c = np.sum(As*taus)



# # TEGDME
# efg_variance = 0.019666563032557 * ke**2 * e**2 / (1e-10)**6  # (V/m)^2
# ### Triexp
# As = np.array([0.31677, 0.04557, 0.63231])
# taus = np.array([0.03658, 2.77812, 177.982])*1e-12
# tau_c = np.sum(As*taus)


print(f"tau_c = {tau_c*1e12:.5f} ps")
CQ = (2*I+3)*(e*Q/hbar)**2 / (I**2 * (2*I-1)) * (1/20)

R1 = CQ * (1+gamma)**2 * efg_variance * tau_c

T1 = 1/R1

print(f" T1 = {T1:.3f}")
