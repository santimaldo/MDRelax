#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:23:15 2023

@author: santi
"""

#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


num_steps = 50000
dt = 10  # fs
times = np.arange(num_steps)*dt

dt_slow = 500
times_slow =  np.arange(0,times[-1]+dt_slow, dt_slow)
Rand_slow = 2 * np.random.rand(times_slow.size) - 1

#Rand_interpolate = interpolate.CubicSpline(times_slow, Rand_slow)
Rand_interpolate = interpolate.interp1d(times_slow, Rand_slow)

Rand_spline= Rand_interpolate(times)
Rand = Rand_spline + (2*np.random.rand(times.size)-1) * 0.1

jj=0

N = int(num_steps/10)
#for ii in range(N):
#    t = times[jj:jj+N]    
#    A = (1+np.random.rand()*0.05) * 10
#    w = (1+np.random.rand()*0.01) * 2*np.pi/(100*N)
#    B = (1+np.random.rand()*0.01) * 1000
    
#    Rand[jj:jj+N] = Rand[jj:jj+N] +  A * np.sin(w*t) + B
#    jj+=N
    
plt.figure(1)
plt.plot(times, Rand_spline, 'b-', markersize=5 )
plt.plot(times, Rand, 'bo')
plt.plot(times_slow, Rand_slow, 'rx', markersize=10)


#%%
tmax = times[-1]
acf = []
taus = []
print(f"Calculating AFC, 00 % progress...")
for ii in range(1,times.size):    
    tau = ii*dt
    jj, t0, acf_ii = 0, 0, 0
    
    if (ii/times.size*100)%10==0:
      print(f"Calculating AFC, {ii/times.size*100:.0f} % progress...")
    while t0+tau<=(tmax):
        #print(f"tau = {tau} ps, t0 = {t0} ps, ---------{jj}")                
        acf_ii += Rand[ii+jj]*Rand[jj]
        jj+=1
        t0 = jj*dt
    # print(f"tau = {tau} ps, t0_max = {t0} fs, ---------")                            
    if jj==0: continue
    #print(f"el promedio es dividir por {jj}")    
    acf.append(acf_ii/jj)
    taus.append(tau)

tau = np.array(taus)
acf = np.array(acf)
tau_c = 80
plt.figure(2)
plt.plot(tau, acf/acf[0], 'o-')
plt.plot(tau, 1.25*np.exp(-tau/tau_c), 'k--')
plt.xlim([0,tau[-1]/2])
plt.show()

datos = np.array([tau[:500], acf[:500]]).T
np.savetxt(f"RandomACF_dtCorr{dt_slow}_Nsteps{num_steps}.dat", datos)
