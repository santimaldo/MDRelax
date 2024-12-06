#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 2024

@author: santi

read mean ACF functions and compare and calculate T1
"""

import matplotlib
matplotlib.use('Qt5Agg')  # Cambia el backend: modo interactivo    
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from Functions import cumulative_simpson
plt.rcParams.update({'font.size': 14})
# Definir función para la suma de N decaimientos exponenciales
def multi_exponential(t, *params):
    N = len(params) // 2
    result = np.zeros_like(t)
    for i in range(N):
        A = params[i]         # Amplitud del i-ésimo término
        tau_i = params[N + i] # Tiempo de decaimiento del i-ésimo término
        result += A * np.exp(-t / tau_i)
    return result

# Parámetros iniciales automáticos para el ajuste
def initial_guess(N, tau):
    amplitudes = [1/N] * N  # Amplitudes iniciales iguales
    times = [0.1] + [(tau[-1] * (i+1) / N) for i in range(1, N)]
    return amplitudes + times



names=[]
paths=[]
# # ACF 0
# names.append("Li+water_guardado-cada-0.001ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/long/")

# # # ACF 1
# names.append("Li+water_guardado-cada-0.01ps")
# paths.append("/home/santi/MD/MDRelax_results/Li-water/freq0.1/")

#############OPLS
# names.append(r"DOL-LiTFSI 0.1 M")
# paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/DOL/run_1ns/")

# names.append(r"DME-LiTFSI 0.1 M")
# paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/DME/run_1ns/")

# names.append(r"Diglyme-LiTFSI 0.1 M")
# paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/Diglyme/run_1ns/")

# names.append(r"TEGDME-LiTFSI 0.1 M")
# paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/TEGDME/run_1ns/")

# names.append(r"ACN-LiTFSI 0.1 M")
# paths.append("/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/ACN/run_1ns/")


names.append(r"DOL-Li$^+$ 0.1 M - charmm36")
paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/DOL_no-anion/")

names.append(r"DME-Li$^+$ 0.1 M - charmm36")
paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/DME_no-anion/")

names.append(r"Diglyme-Li$^+$ 0.1 M - charmm36")
paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/Diglyme_no-anion/")

names.append(r"TEGDME-Li$^+$ 0.1 M - charmm36")
paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/TEGDME_no-anion/")

names.append(r"ACN-Li$^+$ 0.1 M - charmm36")
paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/ACN_no-anion/")


Vsquared_list = []
tau_c_list = []
cutoff_time = 400  # ps
skipdata = 1
N_exp = 3  # Número de exponenciales en la suma
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Proceso de ajuste y gráficos
for idx, (path, name) in enumerate(zip(paths, names)):
    data = np.loadtxt(f"{path}ACF_mean-over-runs.dat")
    tau, ACF = data[::skipdata, 0], data[::skipdata, 1]
    Vsquared_list.append(ACF[0])
    ACF = ACF[tau < cutoff_time]    
    tau = tau[tau < cutoff_time]

    ACF = ACF / max(ACF)  # Normalizar ACF

    # Verificar si "diglyme" o "tegdme" están en "name" (sin distinción de mayúsculas/minúsculas)
    # condicion =  "diglyme" in name.lower() or "tegdme" in name.lower()
    condicion = False
    if condicion:
        # Ajuste usando múltiples exponenciales
        initial_params = initial_guess(N_exp, tau)
        params, _ = curve_fit(multi_exponential, tau, ACF, p0=initial_params)
        ACF_fit = multi_exponential(tau, *params)

        # Extraer amplitudes y tiempos de decaimiento
        amplitudes = np.array(params[:N_exp])
        times = np.array(params[N_exp:])
        tau_c = np.sum(amplitudes*times)
        print(name+f":  tau_c:  {tau_c} ps")


        # Gráfica de ACF y ajuste
        ax = axs[0]
        ax.plot(tau, ACF, 'o-', label=f"{name} ACF", lw=1)
        ax.plot(tau, ACF_fit, 'k--', lw=1)
    else:
        # Solo graficar ACF sin ajuste
        ax = axs[0]
        ax.plot(tau, ACF, 'o-', label=f"{name} ACF", lw=1)
    ax.axhline(0, color='k', ls='--')
    ax.legend(loc="upper right")
    ax.set_ylabel(r"$ACF$", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
    # ax.set_xlim([-0.1,1])

    # Cálculo del tiempo de correlación usando Simpson
    cumulative = cumulative_simpson(ACF, x=tau, initial=0)
    tau_c_list.append(cumulative[-1])
    ax = axs[1]
    ax.plot(tau, cumulative, 'o-', label=name, lw=1)
    # Verificar si "diglyme" o "tegdme" están en "name" (sin distinción de mayúsculas/minúsculas)
    if condicion:
        # Ajuste usando múltiples exponenciales
        cumulative_FIT = cumulative_simpson(ACF_fit, x=tau, initial=0)
        ax.plot(tau, cumulative_FIT, 'k--', lw=1)
    ax.axhline(0, color='k', ls='--')    
    ax.legend()
    ax.set_ylabel(r"$\int_0^{\tau} ACF(t') dt'$  [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)

# Configuración del gráfico final
ax.set_yscale('log')
ax.set_ylim(0.01, 1000)
fig.suptitle("EFG Autocorrelation Function", fontsize=16)
fig.tight_layout()
plt.show()


#%%
### Calculo T1 a partir de DM:
gamma = 0.17  # Sternhemmer factor
# gamma = 0 # Sternhemmer factor
Vsq = np.array(Vsquared_list)
solvents = ["DOL", "DME", "Diglyme","TEGDME", "ACN"]
tau_c = np.array([0.88, 1.5, 9,  72, 0.29])
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

print("T1 by MD:")
for solvent, T1 in zip(solvents, T1_MD):
    print(f"{solvent} :   {T1:.2e} s")
# #%%
# fig,ax = plt.subplots(num=7568756756)
# x = T1_exp
# y = T1_MD
# if gamma==0.17:
#     label = r"Sternheimmer factor: $\gamma_{{\infty}}=$"+f"{gamma}"
# else:
#     fr"Sternheimmer factor: $\gamma={gamma}$"
# ax.scatter(x, y, label=label)

# for i, solvent in enumerate(solvents):
#     ax.annotate(solvent, (x[i], y[i]))
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel(r"$T_{1,exp}$ [s]")
# # ax.set_ylabel(r"$\left(\langle V^2 \rangle \tau_c \right)^{-1}$")
# ax.set_ylabel(r"$T_{1,MD}$ [s]")
# minimo = 0.5*min(min(x),min(y))
# maximo = 2*max(max(x),max(y))
# xx = np.linspace(minimo, maximo,10)
# ax.plot(xx,xx, 'k--', label=r"$T_{1,MD}=T_{1,exp}$" )
# ax.set_xlim([minimo, maximo])
# ax.legend(fontsize=11)
# fig.suptitle(r"LiTFSI 0.1 M - $^7$Li T$_1$")
# # %%

# %%
