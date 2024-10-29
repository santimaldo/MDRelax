#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:44:52 2023

@author: santi
"""
#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

path = "../DATA/2023-12_DME/results/vs_dt/"
filename = f"dt_10fs_ACF-mean.dat"

path = "../DATA/2023-12_TEGDME/results/"
filename = f"ACF-mean.dat"

tau, acf = np.loadtxt(path+filename).T

# Define the weighted sum of two exponential decays
def double_decay(x, a1, tau1, a2, tau2):
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) 

# Generate example data
x_data = tau[tau<21]
y_data = acf[tau<21]

# Initial guess for parameters
initial_guess = [1.0, 0.04, 1.0, 50]
# bounds = ([0, 0, 0, 0],[1.0, 0.05, 1.0, 20])
# Fit the data using curve_fit
params, covariance = curve_fit(double_decay, x_data, y_data, p0=initial_guess)#, bounds=bounds)

# Extract fitted parameters
a1_fit, tau1_fit, a2_fit, tau2_fit = params
# Generate fitted curve
y_fit = double_decay(x_data, a1_fit, tau1_fit, a2_fit, tau2_fit)

# Plot the results
fig, axs= plt.subplots(nrows=1, ncols=2,figsize=(10, 6))
ax = axs[0]
ax.plot(x_data, y_data, 'o-')
ax.plot(x_data, y_fit, linestyle='-', linewidth=2, color='red')
# ax = axs[1]
# ax.plot(x_data, y_data-y_fit, 'o-')
# ax.axhline(0, color='k', ls='--')

ax = axs[1]
ax.loglog(x_data[0:], y_data[0:], 'o-')
ax.loglog(x_data[0:], y_fit[0:], linestyle='-', linewidth=2, color='red')
# xlim = ax.get_xlim()
# ax = axs[1,1]
# ax.plot(x_data[0:], (y_data-y_fit)[0:], 'o-')
# ax.axhline(0, color='k', ls='--')
# ax.set_xscale('symlog')
# ax.set_yscale('symlog')
# ax.set_xlim(xlim)


for ax in axs:
    ax.set_xlabel(r'$\tau$ [ps]')
    ax.set_ylabel('ACF [atomic units]')   

# Display the fitted parameters
print("Fitted Parameters:")
print("a1:", a1_fit)
print("tau1:", tau1_fit)
print("a2:", a2_fit)
print("tau2:", tau2_fit)
print("\n tau_c = c1*tau1+c2*tau2, donde cj = aj/(a1+a2)  :")
print((tau1_fit*a1_fit+tau2_fit*a2_fit)/(a1_fit+a2_fit))















