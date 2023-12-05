#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:00:58 2023

@author: santi

read ACF functions and average
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

frame_times = ["500ps", "1ns"]
runs = ["frames_HQ_1", "frames_HQ_2"]

path = f"../DATA/2023-12_DME/"
plt.figure(0)

suma = 0
for frame_time in frame_times:
  for run in runs:

    
    filename = f"{path}ACF_{frame_time}_{run}.dat"
    
    
    data = np.loadtxt(filename)
    
    plt.plot(data[:,0], data[:,1])
    plt.plot(data[:,0], data[:,2])
    
    suma += data[:,2] + data[:,1]

#%%
plt.figure(30)
plt.plot(data[:,0], suma/suma[0], 'k--')
plt.plot(data[:,0], savgol_filter(suma/suma[0], 500, 3), 'r-', lw=2)

#%%


plt.figure(2)
suma = 0
ACFS = []
dt = times[1]-times[0]
for frame_time in frame_times:
  for run in runs:
    
    filename = f"{path}ACF_{frame_time}_{run}.dat"    
    data = np.loadtxt(filename)
    
    for nn in range(1,3):
        plt.plot(data[:,0], data[:,nn])
        t = data[:,0]
        
        acf = np.zeros([t.size])
        Num_promedios = np.zeros(t.size)
        for ii in range(t.size):    
            tau = ii*dt
            jj, t0, acf_ii = 0, 0, 0
            while t0+tau<=times[-1]:
                print(f"tau = {tau} fs, t0 = {t0} ps, ---------{jj}")                
                acf_ii += data[ii,nn]*data[ii+jj,nn]
                jj+=1
                t0 = t0+dt
            print(f"el promedio es dividir por {jj}")
            acf[ii] = acf_ii/jj
            Num_promedios[ii] = jj
        ACFS.append(acf)
        

#%%
ACF = np.zeros_like(acf)      
for n in range(len(ACFS)):
    ACF +=  ACFS[n]
    
plt.plot(t, ACF/ACF[0], 'Gray', ls='--')
plt.plot(t, savgol_filter(ACF/ACF[0], 500, 3), 'b-', lw=2)