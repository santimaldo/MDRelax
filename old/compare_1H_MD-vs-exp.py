#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 12  2024

@author: santi

Comparo T1 de 1H experimental vs simulacion
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


solvents = ["DOL", "DME", "Diglyme" , "TEGDME", "ACN"]

# valores experimentales:
# neat
T1_exp = [11.74,9.66, 4.50, 1.45, 16.5] 
T1_err = [0.13, 0.23, 0.05, 0.05, 0.1]
# with LiTFSI
# T1_exp = [11.74,8.53, 4.04, 1.34, 13.33] 
# T1_err = [0.09, 0.18, 0.05, 0.04, 0.62]

T1_exp = np.array(T1_exp)

### Calculo T1 a partir de DM:
T1_MD = [8.80, 4.89,0.954, 0.105, 16.8]


T1_exp_charmm = [11.74, 1.45, 9.66, 4.50, 16.5] 
T1_MD_charmm = [13.2, 1.1, 9.25, 2.15, 31.7]


T1_exp_charmm_all_proton = [11.74, 9.66, 4.50, 16.5]
T1_MD_charmm_all_proton = [11.8, 9.84, 2.41,28.1]

#%%
fig,ax = plt.subplots(num=75687567575867566)
x = T1_exp
y = T1_MD
label = r"OPLS"
ax.scatter(x, y, label=label)

for i, solvent in enumerate(solvents):
    ax.annotate(solvent, (x[i], y[i]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$T_{1,exp}$ [s]")
# ax.set_ylabel(r"$\left(\langle V^2 \rangle \tau_c \right)^{-1}$")
ax.set_ylabel(r"$T_{1,MD}$ [s]")
minimo = 0.5*min(min(x),min(y))
maximo = 2*max(max(x),max(y))
xx = np.linspace(minimo, maximo,10)
ax.plot(xx,xx, 'k--', label=r"$T_{1,MD}=T_{1,exp}$" )
ax.set_xlim([minimo, maximo])

x = T1_exp_charmm
y = T1_MD_charmm
label = r"CHARMM36"
ax.scatter(x, y, label=label)

x = T1_exp_charmm_all_proton
y = T1_MD_charmm_all_proton
label = r"CHARMM36 - T1-ALL-PROTON"
ax.scatter(x, y, label=label, marker='x', color='red')



x = [2.836]
y = [3.3]
label = r"TIP4-2005"
solvents = ["Water"]
ax.scatter(x, y, label=label, marker='x', color='g')
for i, solvent in enumerate(solvents):
    ax.annotate(solvent, (x[i], y[i]))

ax.legend(fontsize=11)
fig.suptitle(r"LiTFSI 0.1 M - $^1$H T$_1$")
# %%