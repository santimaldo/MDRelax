#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:37:52 2023

@author: santi
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})

path = "../DATA/2023-12_DME/results/vs_dt/"

dts = [10,20,50,100,500,1000]


fig, axs =plt.subplots(num=1, nrows=1, ncols=2, figsize=(9,6))
figC, axC = plt.subplots(num=2)
for dt in dts:
    
    filename = f"dt_{dt}fs_ACF-mean.dat"
    tau, acf = np.loadtxt(path+filename).T
    # PASO ACF A UNIDADES ATOMICAS:
    # 1 Ang = 1.88973 bohr radius
    factor = (1.88973)**6
    acf = acf/factor
    
    y_data = acf[tau<22]
    x_data = tau[tau<22]
    
    # Calculate the Fourier transform
    # fft_result = np.fft.fft(y_data)
    # frequencies = np.fft.fftfreq(len(x_data), (x_data[1] - x_data[0]))
    # frequencies = np.fft.fftshift(frequencies)
    # fft_result = np.fft.fftshift(np.fft.fft(y_data))
    
    ms = (np.log(dt)/2)**2
    ax =axs[0]
    ax.plot(x_data, y_data, 'o-', label=f'dt = {dt} fs', ms=1)    

    ax = axs[1]
    ax.plot(x_data, y_data, 'o-', label=f'dt = {dt} fs', ms=ms)
    ax.set_xlim([-0.05,1.05])
    
    
    # ax = axs[2]
    # ax.plot(x_data, y_data, 'o-', label=f'dt = {dt} fs', ms=ms)    
    # ax.set_xlim([-0.02,0.45])
    # ax.set_ylim([0.01/factor,0.021/factor])
    
    filename = f"dt_{dt}fs_Cumulative-Mean.dat"
    tau, Ct = np.loadtxt(path+filename).T
    
    
    axC.plot(tau[1:], Ct[1:], 'o-', label=f'dt = {dt} fs', ms=2)    

axC.set_xlabel(r'$\tau$ [ps]')
axC.set_ylabel('Cumulative Integra [ps]')
axC.legend()    
    
    
for ax in axs:
    ax.set_xlabel(r'$\tau$ [ps]')
    ax.set_ylabel('ACF [atomic units]')    
axs[0].legend()
axs[1].legend(loc="lower left")
# axs[1].legend()
fig.tight_layout()

