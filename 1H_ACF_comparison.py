#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 2024

@author: santi

read mean ACF functions and compare and calculate T1
"""
    
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


solvents = ["DOL", "DME", "Diglyme" , "TEGDME", "ACN"]


for solvent in solvents:
    names.append(f"{solvent}-LiTFSI 0.1 M")
    paths.append(f"/home/santi/MD/MDRelax_results/LiTFSI_small-boxes/1H/tmp/1H_ACF_{solvent}.dat")

Vsquared_list = []
tau_c_list = []
cutoff_time = 1  # ps
skipdata = 1
N_exp = 4  # Número de exponenciales en la suma
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

# Proceso de ajuste y gráficos
for idx, (path, name) in enumerate(zip(paths, names)):
    data = np.loadtxt(f"{path}")
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
fig.suptitle("Dipole-Dipole Autocorrelation Function", fontsize=16)
fig.tight_layout()
plt.show()
