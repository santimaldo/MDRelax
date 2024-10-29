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
plt.rcParams.update({'font.size': 16})

# Define the weighted sum of two exponential decays
def ExpDecN(x, a1, tau1, a2, tau2):
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) 

# Define the weighted sum of N exponential decays
def ExpDecN(x, *params):
    y = np.zeros_like(x)
    N = len(params) // 2
    for i in range(N):
        a = params[2*i]
        tau = params[2*i + 1]
        y += a * np.exp(-x / tau)
    return y


names=[]
paths=[]
# ACF 0
# names.append("DME-LiTFSI")
# paths.append("/home/santi/MD/MDRelax_results/DME_LiTFSI/")

# ACF 1
# names.append(r"DME-$Li_2S_6$")
# paths.append("/home/santi/MD/MDRelax_results/DME_PS/")

# ACF 2
# names.append(r"DME-$Li^+$")
# paths.append("/home/santi/MD/MDRelax_results/DME_no-anion/")

names.append("DME-Li+: UHQ")
paths.append("/home/santi/MD/MDRelax_results/DME_no-anion_bigbox_UHQ/upto5ps/")

# numero de exponenciales
# tau_guess = [0.03,0.05,5,20]
# tau_guess = [0.03,1,5,10]
tau_guess = [0.05,5,50]
A_j_guess = [0.3,0.3,0.3]


# processing parameters:
# from cut-off time, the acf is forced to zero
cutoff_time = 80 # ps


colors_data = ['cornflowerblue', 'lightcoral', 'lightgreen']
colors_fit = ['midnightblue', 'firebrick', 'forestgreen']
efg_variances = []
correlation_times = []
N = len(tau_guess)
# fig, ax= plt.subplots(nrows=1, ncols=1,figsize=(10, 6))
fig, (ax, ax_resid) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
for idx, (path, name) in enumerate(zip(paths, names)):

    data = np.loadtxt(f"{path}ACF_mean-over-runs.dat")
    tau, acf = data[:,0], data[:,1]

    acf_full = acf
    tau_full = tau

    acf_excluded = acf[tau>cutoff_time]
    tau_excluded = tau[tau>cutoff_time]    

    acf = acf[tau<=cutoff_time]
    tau = tau[tau<=cutoff_time]

    # Initial guess for parameters (a_j and tau_j for j=1 to N)
    initial_guess = np.zeros(2*N)
    initial_guess[::2]  = np.array(A_j_guess)*acf[0]
    initial_guess[1::2] = tau_guess
    bounds = ([0]*(2*N), [1,1000]*N)

    x_data, y_data = tau, acf
    # Fit the data using curve_fit
    params, covariance = curve_fit(ExpDecN, x_data, y_data,
                                   p0=initial_guess, 
                                   bounds=bounds,
                                   method='trf')
    
    # Generate fitted curve
    y_fit = ExpDecN(x_data, *params)

    # Calculate weighted average decay time tau_c
    tau_c = sum(params[2*i] * params[2*i + 1] for i in range(N)) / sum(params[2*i] for i in range(N))
    correlation_times.append(tau_c)
    efg_variances.append(np.loadtxt(f"{path}EFG_variance_mean-over-runs.dat"))
    # Plot the results    
    # ax = axs[0]
    label = name+"\n"\
            r"$\tau_c\equiv $"\
            r"$\left(\sum_j A_j \tau_j \right)$"\
            r"$\left(\sum_j A_j\right)^{-1} = $"\
            f" {tau_c:.2f} ps"    
    ax.plot(tau_excluded, acf_excluded, 'o', mfc='w', mec=colors_data[idx], alpha=0.05)
    ax.plot(tau_excluded, ExpDecN(tau_excluded, *params),
            linestyle='--', linewidth=1, color=colors_fit[idx])
    ax.plot(x_data, y_data, 'o', label=label, color=colors_data[idx])
    ax.plot(x_data, y_fit, linestyle='-', linewidth=2, color=colors_fit[idx])
    
    # Plot the residuals (Observed - Fitted)
    # Calculate Rsqares
    residuals = y_data - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)    
    label=f"{name}, "+r"$R^2$"+f" = {r_squared:.5f}"
    ax_resid.plot(x_data, residuals, 'o-', color=colors_data[idx], label=label)
    
    
    # Display the fitted parameters
    print("Fitted Parameters:")
    print(f"R^2:\n\t", f"{r_squared:.5f}")
    print("tau_j:\n\t", [f"{p:.2f}" for p in params[1::2]], "  ps")
    # print("A_j:\n\t", [f"{p:.2f}" for p in params[::2]])
    tot = np.sum(params[::2])
    print("c_j=A_j/sum A_j:\n\t", [f"{p/tot:.2f}" for p in params[::2]])            
    print("tau_c = sum_j c_j*tau_j:\n\t", f"{tau_c:.2f} ps")    

# Formatting fit plot
# ax.set_xlim([-0.1*x_data[-1], 1.3*x_data[-1]])
ax.set_xlabel(r'$\tau$ [ps]')
ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$")
ax.axhline(0, color='black', linestyle='--', linewidth=1)
legend_title = r"$ACF(\tau) = \sum_j A_j \exp^{-\tau/\tau_j}$"    
ax.legend(title=legend_title, fontsize=14)


### tmp settings
ax.set_xlim([-0.1, 1])
# ax.set_ylim([1e-4,3e-2])
# ax.set_xscale('log')
# ax.set_yscale('log')

# Formatting residuals plot
ax_resid.axhline(0, color='k', ls='--')
ax_resid.set_xlim(ax.get_xlim())
ax_resid.set_xlabel(r'$\tau$ [ps]')
ax_resid.set_ylabel('Residuals')
ax_resid.axhline(0, color='black', linestyle='--', linewidth=1)
ax_resid.legend(fontsize=12)

fig.tight_layout()

#%% Calculating T1:

e = 1.60217663 * 1e-19  # Coulomb
hbar = 1.054571817 * 1e-34  # joule seconds
ke = 8.9875517923 * 1e9  # Vm/C, Coulomb constant

Q = -4.01 * (1e-15)**2  # m**2
I = 1.5  # spin 3/2

gamma = 0.17  # Sternheimmer factor

for idx in range(len(correlation_times)):

    efg_variance = efg_variances[idx]* ke**2 * e**2 / (1e-10)**6 # (V/m)^2
    tau_c = correlation_times[idx]*1e-12 # seconds

    
    CQ = (2*I+3)*(e*Q/hbar)**2 / (I**2 * (2*I-1)) * (1/20)
    R1 = CQ * (1+gamma)**2 * efg_variance * tau_c
    T1 = 1/R1
    
    print(f"{names[idx]}:")    
    print(f"\t <V^2> = {efg_variances[idx]:.3e} e^2 A^-6 (4pi epsilon0)^-2")
    print(f"\t tau_c = {tau_c*1e12:.1f} ps")
    print(f"\t T1 = {T1:.3f} s")

# %%
