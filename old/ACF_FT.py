#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 2024

@author: santi

read mean ACF functions and calculate Fourier Transform
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft, fftfreq, fftshift

path_MDrelax = path = "/home/santi/MD/MDRelax_results/Li-water/long/"
path = path_MDrelax

# from cut-off time, the acf is forced to zero
cutoff_time = 20 # ps


# tau0, acf0 = np.loadtxt(f"{path}/ACF_mean_over_runs.dat").T
data = np.loadtxt(f"{path}/ACF_HQ.6000_ps.long.dat")
tau0 = data[:2**18,0]
acf0 = data[:2**18,1]
del data
acf0 = acf0[tau0<cutoff_time]
tau0 = tau0[tau0<cutoff_time]

acf0[tau0>cutoff_time] = 0
N = 2**(math.ceil(np.log2(acf0.size)))
N = 2**18

tau = np.arange(N)*[tau0[1]-tau0[0]]
acf = np.zeros(N)
acf[:acf0.size] = acf0[:acf0.size]

plt.figure(1)
plt.plot(tau, acf)
plt.plot(tau0, acf0)


### FFT:
Npts = tau.size
dw = (tau[1]-tau[0])*1e-12 # seconds
spectral_density = fftshift(fft(acf))
freq = fftshift(fftfreq(Npts, dw))

plt.figure(2)
plt.semilogy(freq, spectral_density.real)
plt.xlabel(r"$\omega$ [$s^{-1}$]")
plt.ylabel(r"$J(\omega)$")


#%%
plt.figure(3)
plt.semilogx(freq, spectral_density.real, 'k-')
plt.semilogx(freq, spectral_density.real, 'o', ms=2)
plt.xlabel(r"$\omega$ [$s^{-1}$]")
plt.ylabel(r"$J(\omega)$")

"""
La figura 3 me hace pensar que es correcta la aproximacion de
utilizar el limite w->0
"""

# %%
