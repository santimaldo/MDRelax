#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 2024

@author: santi

read mean ACF functions and compare and calculate T1
"""
    

import matplotlib.pyplot as plt
import numpy as np
from Functions import cumulative_simpson
plt.rcParams.update({'font.size': 16})


names=[]
paths=[]
# ACF 0
# names.append("Li+water_guardado-cada-0.001ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/long/")

# # ACF 1
# names.append("Li+water_guardado-cada-0.01ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/freq0.1/")

# ACF 0
names.append(r"Water-$Li^+$")
paths.append("/home/santi/MD/MDRelax_results/Li-water/long/")
# ACF 2
names.append(r"DME-$Li^+$")
paths.append("/home/santi/MD/MDRelax_results/DME_small-boxes/DME_no-anion/")

cutoff_time = 100 # ps
fig, axs= plt.subplots(nrows=2, ncols=1,figsize=(10, 10))
# fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
for idx, (path, name) in enumerate(zip(paths, names)):

    data = np.loadtxt(f"{path}ACF_mean-over-runs.dat")
    tau, ACF = data[:,0], data[:,1]
    del data

    ACF = ACF[tau<cutoff_time]
    tau = tau[tau<cutoff_time]    


    # FIGURA: Autocorrelaciones        
    ACF = ACF/max(ACF)
    ax = axs[0]
    ax.plot(tau, ACF, 'o-', label=name, lw=1)
    ax.axhline(0, color='k', ls='--')
    ax.legend()
    ax.set_ylabel(r"$ACF$", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    


    cumulative = cumulative_simpson(ACF, x=tau, initial=0)
    ax = axs[1]
    ax.plot(tau, cumulative, 'o-', label=name, lw=1)
    ax.axhline(0, color='k', ls='--')
    ax.legend()    
    ax.set_ylabel(r"$\int_0^{\tau} ACF(t') dt'$  [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
ax.set_yscale('log')
ax.set_ylim(0.01,10)
fig.suptitle(fr"EFG Autocorrelation Function", fontsize=16)
fig.tight_layout()
