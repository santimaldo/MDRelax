#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 12  2024

@author: santi

Comparo T1 de 7Li vs 1H en experimental y simulacion
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


solvents = ["DOL", "DME", "Diglyme" , "TEGDME", "ACN"]
### OPLS
T1_MD = np.array([3.92297875e+00, 1.26335644e+00, 2.79360592e-01, 3.06262187e-02, 5.95360631e+01])
T1_exp = np.array([13.24 , 10.34 ,  3.66 ,  0.708, 24.9  ])


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
ax.legend(fontsize=11)
fig.suptitle(r"LiTFSI 0.1 M - T$_1$ - MD vs Experimental")


solvents = ["DOL", "DME", "Diglyme","TEGDME", "ACN"]
### CHARMM
T1_exp = np.array([13.24 , 10.34, 3.66,0.708, 24.9])
T1_MD = np.array([18.5, 12.5, 1.85,0.202, 73.3])
T1_MD = np.array([20.8, 12.1, 2.07, 0.21, 73.1])
x = T1_exp
y = T1_MD
label = r"CHARMM"
ax.scatter(x, y, label=label)
# for i, solvent in enumerate(solvents):
#     ax.annotate(solvent, (x[i], y[i]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()


# %%
