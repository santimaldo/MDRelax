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

# valores experimentales:
# neat
T1_exp_1H = np.array([11.74,9.66, 4.50, 1.45, 14.51])
# T1_err = [0.13, 0.23, 0.05, 0.05, 0.1]
# with LiTFSI
# T1_exp = [11.74,8.53, 4.04, 1.34, 13.33] 
# T1_err = [0.09, 0.18, 0.05, 0.04, 0.62]
### Calculo T1 a partir de DM:
T1_MD_1H = [8.80, 4.89,0.954, 0.105, 16.8]

#==============================================

T1_MD_7Li = np.array([3.92297875e+00, 1.26335644e+00, 2.79360592e-01, 3.06262187e-02, 5.95360631e+01])
T1_exp_7Li = np.array([13.24 , 10.34 ,  3.66 ,  0.708, 24.9  ])
#%%
fig,ax = plt.subplots(num=75687567575867566)
x = T1_exp_1H
y = T1_exp_7Li
label = r""
ax.scatter(x, y, label=label)
for i, solvent in enumerate(solvents):
    ax.annotate(solvent, (x[i], y[i]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$T_{1,^1H}$ [s]")
# ax.set_ylabel(r"$\left(\langle V^2 \rangle \tau_c \right)^{-1}$")
ax.set_ylabel(r"$T_{1,^7Li}$ [s]")
minimo = 0.5*min(min(x),min(y))
maximo = 2*max(max(x),max(y))
xx = np.linspace(minimo, maximo,10)
ax.plot(xx,xx, 'k--', label=r"$T_{1,^7Li}=T_{1,^1H}$" )
ax.set_xlim([minimo, maximo])
ax.legend(fontsize=11)
fig.suptitle(r"LiTFSI 0.1 M - T$_1$ - Experimental")
#%%
fig,ax = plt.subplots(num=75687567575867567)
x = T1_MD_1H
y = T1_MD_7Li
label = r""
ax.scatter(x, y, label=label)
for i, solvent in enumerate(solvents):
    ax.annotate(solvent, (x[i], y[i]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$T_{1,^1H}$ [s]")
# ax.set_ylabel(r"$\left(\langle V^2 \rangle \tau_c \right)^{-1}$")
ax.set_ylabel(r"$T_{1,^7Li}$ [s]")
minimo = 0.5*min(min(x),min(y))
maximo = 2*max(max(x),max(y))
xx = np.linspace(minimo, maximo,10)
ax.plot(xx,xx, 'k--', label=r"$T_{1,^7Li}=T_{1,^1H}$" )
ax.set_xlim([minimo, maximo])
ax.legend(fontsize=11)
fig.suptitle(r"LiTFSI 0.1 M - T$_1$ - Molecular Dynamics")

#%%
fig,axs = plt.subplots(num=75687567485867568, nrows=2)

y = T1_exp_1H/T1_exp_7Li
ax = axs[0]
ax.set_title("Experimental")
ax.bar(solvents, y)
ax.set_ylabel(r"$T_{1,^1H}/T_{1,^7Li}$")

y = T1_MD_1H/T1_MD_7Li
ax = axs[1]
ax.set_title("Molecular Dynamics")
ax.bar(solvents, y)
ax.set_ylabel(r"$T_{1,^1H}/T_{1,^7Li}$")

fig.tight_layout()

# %%
