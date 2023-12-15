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
### Metodo integral:
tau_c = 5.6 * 1e-12 # s
As = np.array([0.34845, 0.13996, 0.49357])
taus = np.array([0.03717, 1.01674, 11.13488])*1e-12
tau_c = np.sum(As*taus)/np.sum(As)
efg_variance = 0.0206515220481938* ke**2 * e**2 / (1e-10)**6 # (V/m)^2

tau_c = 5.6 * 1e-12 # s
efg_variance = 0.0207* ke**2 * e**2 / (1e-10)**6 # (V/m)^2

# TEGDME
# efg_variance = 0.019666563032557 * ke**2 * e**2 / (1e-10)**6  # (V/m)^2
### Sin quitar nada
# A1 = 0.285
# A2 = 0.651
# t1 = 0.0692
# t2 = 130.87
### Quitando un poco
# A1 = 0.36774
# A2 = 0.54
# t1 = 0.033329
# t2 = 133.8
### Quitando Mucho
# A1 = 0.3841
# A2 = 0.6386
# t1 = 0.03181
# t2 = 160.894
### finalmente:
# tau_c = (A1*t1+A2*t2)/(A1+A2) * 1e-12  # s
### Triexp
# As = np.array([0.31677, 0.04557, 0.63231])
# taus = np.array([0.03658, 2.77812, 177.982])*1e-12
# tau_c = np.sum(As*taus)/np.sum(As)





print(f"tau_c = {tau_c*1e12:.5f} ps")
CQ = (2*I+3)*(e*Q/hbar)**2 / (I**2 * (2*I-1)) * (1/20)

R1 = CQ * (1+gamma)**2 * efg_variance * tau_c

T1 = 1/R1

print(f" T1 = {T1:.3f}")
