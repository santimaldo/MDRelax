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
# # ACF 0
# names.append("Li+water_guardado-cada-0.001ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/long/")

# # # ACF 1
# names.append("Li+water_guardado-cada-0.01ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/freq0.1/")

#ACF 0
names.append(r"DME-LiTFSI 0.1 M")
paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/DME/run_1ns/")
# ACF 1
names.append(r"TEGDME-LiTFSI 0.1 M")
paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/TEGDME/run_1ns/")
# ACF 2
names.append(r"ACN-LiTFSI 0.1 M")
paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/ACN/run_1ns/")


Vsquared_list = []
tau_c_list = []
cutoff_time = 10# ps
skipdata = 1
fig, axs= plt.subplots(nrows=2, ncols=1,figsize=(10, 10))
# fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
for idx, (path, name) in enumerate(zip(paths, names)):

    data = np.loadtxt(f"{path}ACF_mean-over-runs.dat")
    tau, ACF = data[::skipdata,0], data[::skipdata,1]
    del data
    Vsquared_list.append(ACF[0])
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
    tau_c_list.append(cumulative[-1])
    ax = axs[1]
    ax.plot(tau, cumulative, 'o-', label=name, lw=1)
    ax.axhline(0, color='k', ls='--')
    ax.legend()    
    ax.set_ylabel(r"$\int_0^{\tau} ACF(t') dt'$  [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
ax.set_yscale('log')
ax.set_ylim(0.01,1000)
fig.suptitle(fr"EFG Autocorrelation Function", fontsize=16)
fig.tight_layout()



#%%

solvents = ["DME", "TEGDME", "ACN"]
T1_exp = [10.34, 0.708, 24.9]
T1_err = [0.81, 0.03, 5]

T1_exp = np.array(T1_exp)

gamma = 0.17  # Sternhemmer factor
# gamma = 0 # Sternhemmer factor
Vsq = np.array(Vsquared_list)
tau_c = np.array([5.843533266714134 ,3.07493696e+02 , 0.22])
e = 1.60217663 * 1e-19  # Coulomb
hbar = 1.054571817 * 1e-34  # joule seconds
ke = 8.9875517923 * 1e9  # Vm/C, Coulomb constant
Q = -4.01 * (1e-15)**2  # m**2
I = 1.5  # spin 3/2
# water
efg_variance = Vsq* ke**2 * e**2 / (1e-10)**6 # (V/m)^2
CQ = (2*I+3)*(e*Q/hbar)**2 / (I**2 * (2*I-1)) * (1/20)
R1 = CQ * (1+gamma)**2 * efg_variance * (tau_c*1e-12)
T1_MD = 1/R1


fig,ax = plt.subplots(num=7568756756)
x = T1_exp
y = T1_MD
if gamma==0.17:
    label = r"Sternheimmer factor: $\gamma_{{\infty}}=$"+f"{gamma}"
else:
    fr"Sternheimmer factor: $\gamma={gamma}$"
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
fig.suptitle("LiTFSI 0.1 M")
# %%
