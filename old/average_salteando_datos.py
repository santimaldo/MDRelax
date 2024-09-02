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
from scipy.integrate import cumulative_trapezoid
plt.rcParams.update({'font.size': 12})
#%%
frame_times = ["500ps", "1ns"]
runs = ["frames_HQ_1", "frames_HQ_2"]

# frame_times = ["500ps"]
# runs = ["frames_HQ_1"]
solvent = "DME"

path = f"../DATA/2023-12_{solvent}/results/"

# Number of time steps
dt_new = 10 # fs 
dt_true = 10 # fs
Nskip = int(dt_new/dt_true)
times = np.arange(0,30.001,step=dt_new/1000)
Ntimes = times.size
# Number of runs
Nruns = len(frame_times)*len(runs)
# Number of Li ions in a run 
NLi = 2


acf_data = np.zeros([Ntimes, Nruns, NLi])
efg_variance = np.zeros([Nruns, NLi])
run_ind = 0
# Loop sobre runs para calcular ACF
for frame_time in frame_times:
  for run in runs:
    
    filename = f"{path}/EFG_{frame_time}_{run}.dat"    
    data_full = np.loadtxt(filename)
    data = data_full[::Nskip, :]
    # data columns order:    
    # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
    # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
    for nn in range(2): # 2, uno para cada litio
        # plt.plot(data[:,0], data[:,nn+1])
        t = data[:,0]
        Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
        Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]
        
        if run_ind + nn == 0:        
            plt.figure(1)
            plt.plot(t, np.array([Vxx, Vyy, Vzz, Vxy, Vyz, Vxz]).T)                
            plt.plot(t, Vxx+Vyy+Vzz, 'k', label='Trace')                
            plt.legend(["Vxx", "Vyy", "Vzz", "Vxy", "Vyz", "Vxz", "Trace: Vxx+Vyy+Vzz"],
                        ncols=4, fontsize=10)
            plt.title(f"{solvent};   run:{frame_time},{run};   Li{nn+1}\n dt = {dt_new} fs")
            plt.xlabel("Time [ps]")
            plt.ylabel(r"EFG [$e/\AA^3$]")
            plt.tight_layout()        
            plt.savefig(f"{path}/vs_dt/Figuras/dt_{dt_new}fs_EFG_{frame_time}_{run}_Li{nn+1}.png")
            plt.close()
    
        # Calculo ACF
        Ntau = t.size                               
        tau = np.zeros([Ntau])
        acf = np.zeros([Ntau])        
        efg_squared = np.zeros([Ntau])        
        dt = t[1]-t[0]
        for jj in range(Ntau):    
            tau_jj = jj*dt            
            max_tau_index = t.size-jj            
            # jj, t0, acf_ii = 0, 0, 0            
            # while t0+tau<=times[-1]:
            acf_jj = 0     
            if jj%500==0:                       
                print(f"RUN {run_ind},  Li{nn+1},  tau = {tau_jj:.2f}")
            for ii in range(0,max_tau_index):
                # if ii%100==0:
                #     print(f"tau = {tau_jj:.2f} fs, t0 = {ii*dt:.2f} ps,"\
                #           f" EFG[{ii}]*EFG[{ii+jj}]")                
                product = 0
                product += Vxx[ii]*Vxx[ii+jj] 
                product += Vyy[ii]*Vyy[ii+jj] 
                product += Vzz[ii]*Vzz[ii+jj]
                product += 2 * Vxy[ii]*Vxy[ii+jj]
                product += 2 * Vyz[ii]*Vyz[ii+jj]
                product += 2 * Vxz[ii]*Vxz[ii+jj]
                acf_jj += product
                                                
            tau[jj] = tau_jj
            promedio = acf_jj/(max_tau_index+1)
            acf[jj] = promedio
            
                        
            acf_data[:, run_ind, nn] = acf
            
            efg_squared = Vxx**2+Vyy**2+Vzz**2 + 2*(Vxy**2+Vyz**2+Vxz**2)
            efg_variance[run_ind, nn] = np.mean(efg_squared)
    run_ind += 1
#%%

# graficos:
fignum = 2
fig = plt.figure(num=fignum, figsize=(12, 8))
colors = ['k', 'b', 'r', 'g']
run_ind = -1
ii, jj = -1,-1
for frame_time in frame_times:
  ii+=1
  for run in runs:
      jj+=1
      run_ind += 1
      jjcond = jj
      if ii>0:
          jjcond = jj-2
      ax = plt.subplot2grid((int(Nruns/2),Nruns), (ii, jjcond))
      ax.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])
      if jj%2==1:
          ax.yaxis.set_ticklabels([])
      else:
          ax.set_ylabel(r"ACF $[e^2/\AA^3]$", fontsize=16)
      
      ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
      for nn in range(2): # 2, uno para cada litio
        acf = acf_data[:,run_ind, nn]
        alpha = 1
        promedio_entre_litios = 0
        if nn>0: alpha=0.5
        ax.plot(tau, acf, label=f"Li{nn+1}", lw=2,
                 color=colors[run_ind], alpha=alpha)                    
      ax.axhline(0, color='k', ls='--')
      ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")

      
ax = plt.subplot2grid((int(Nruns/2),Nruns), (0,int(Nruns/2)),
                      rowspan=int(Nruns/2),colspan=int(Nruns/2))

promedio = np.mean(acf_data, axis=(1,2))
datos = np.array([tau, promedio]).T
np.savetxt(f"{path}/vs_dt/dt_{dt_new}fs_ACF-mean.dat", datos)

label = "Mean ACF"
ax.plot(tau, promedio, 'o-', color='orange', label=label, lw=1)        
ax.axhline(0, color='k', ls='--')
ax.set_ylabel(r"ACF $[e^2/\AA^3]$", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.legend()
enter = "\n"
fig.suptitle(fr"{solvent}$-Li_2S_6$ EFG Autocorrelation Function{enter} dt = {dt_new} fs", fontsize=22)
fig.tight_layout()
fig.savefig(f"{path}/vs_dt/Figuras/dt_{dt_new}fs_ACF.png")

#%%
# graficos:
fignum = 3
fig = plt.figure(num=fignum, figsize=(12, 8))
colors = ['k', 'b', 'r', 'g']
run_ind = -1
ii, jj = -1,-1

cumulative_data = np.zeros_like(acf_data)
for frame_time in frame_times:
  ii+=1
  for run in runs:
      jj+=1
      run_ind += 1
      jjcond = jj
      if ii>0:
          jjcond = jj-2
      ax = plt.subplot2grid((int(Nruns/2),Nruns), (ii, jjcond))
      # ax.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])
      if jj%2==1:
          ax.yaxis.set_ticklabels([])
      else:
          ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
      
      ax.set_xlabel(r"$C(\tau)$ [ps]", fontsize=16)
      for nn in range(2): # 2, uno para cada litio
        acf = acf_data[:,run_ind, nn]
        alpha = 1
        promedio_entre_litios = 0
        if nn>0: alpha=0.5
        
        integral = cumulative_trapezoid(acf, x=tau, initial=0)
        cumulative = integral/efg_variance[run_ind, nn]        
        ax.plot(tau, cumulative, label=f"Li{nn+1}", lw=2,
                 color=colors[run_ind], alpha=alpha)           
        # guardo los datos para luego promediar
        cumulative_data[:, run_ind, nn] = cumulative
      ax.axhline(0, color='k', ls='--')
      ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")      
      
      
      
ax = plt.subplot2grid((int(Nruns/2),Nruns), (0,int(Nruns/2)),
                      rowspan=int(Nruns/2),colspan=int(Nruns/2))
cumulative_promedio = np.mean(cumulative_data, axis=(1,2))
datos = np.array([tau, cumulative_promedio]).T
np.savetxt(f"{path}/vs_dt/dt_{dt_new}fs_Cumulative-Mean.dat", datos)

label = "Mean Cumulative"
ax.plot(tau, cumulative_promedio, 'o-', color='orange', label=label, lw=1)        
ax.axhline(0, color='k', ls='--')
ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.legend()
title = f"{solvent}"\
        r"$-Li_2S_6$,"\
        f"   dt = {dt_new} fs \n"\
        r" Cumulative Integral of ACF:   "\
        r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
        r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
        r"$V_{\alpha\beta}\rangle(t') dt'$"
fig.suptitle(title, fontsize=18)
fig.tight_layout()
fig.savefig(f"{path}/vs_dt/Figuras/dt_{dt_new}fs_CorrelationTime.png")





# #%%
# fignum =30
# fig, axs = plt.subplots(num=fignum, nrows=1, ncols=2, figsize=(10, 6))

# ax = axs[0]
# label = "Promedio sobre el total\n de trayectorias y litios"
# ax.plot(tau, promedio/Nruns, color='yellow', label=label, lw=3)        
# label = "Promedio sobre trayectorias\n del promedio sobre litios"
# ax.plot(tau, promedios_promedios*2/run_ind, color='orange', label=label, lw=3)        
# ax.axhline(0, color='k', ls='--')
# ax.set_ylabel("ACF", fontsize=16)
# ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
# ax.legend()
# # ---------------------------------------
# ax = axs[1]
# label = "Promedio sobre el total\n de trayectorias y litios"
# integral0 = cumulative_trapezoid(promedio/Nruns, x=tau, initial=0)
# ax.plot(tau, integral0, color='yellow', label=label, lw=3)        

# label = "Promedio sobre trayectorias\n del promedio sobre litios"
# integral1 = cumulative_trapezoid(promedios_promedios*2/run_ind, x=tau, initial=0)
# ax.plot(tau, integral1, color='orange', label=label, lw=3)        

# ax.axhline(0, color='k', ls='--')
# ax.set_ylabel("ACF", fontsize=16)
# ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
# ax.yaxis.set_label_position("right")
# ax.yaxis.tick_right()
# ax.legend()


# fig.suptitle(fr"{solvent}$-Li_2S_6$", fontsize=22)
# fig.tight_layout()

# # s0, s1 = savgol_filter(suma[:,0], 500, 3), savgol_filter(suma[:,1], 500, 3)
# # # s0, s1 = suma[:,0], suma[:,1]

# # integral0, integral1 = cumulative_trapezoid(s0, x=t, initial=0), cumulative_trapezoid(s1, x=t, initial=0)

# # plt.figure(10)
# # plt.plot(t, integral0)
# # plt.plot(t, integral1)
