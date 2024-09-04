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
from scipy.integrate import cumulative_trapezoid, simpson
import time
plt.rcParams.update({'font.size': 12})

#%%
def cumulative_simpson(ydata, x=None, initial=0):
    """
    Compute cumulative integra with simpson's rule
    ydata must be a 1D array
    """
    # inicializo    
    if x is None: x=np.arange(ydata.size)        
    integral = np.zeros_like(ydata)
    integral[0] = initial
    for nf in range(1, ydata.size):         
        ytmp, xtmp = ydata[:nf+1], x[:nf+1]    
        integral[nf] = simpson(ytmp, x=xtmp)    
    return integral
        
    
    

#%%

path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/MDRelax/"
cation, anion, solvent_hr = ["Li", "S6", "DME"] # hr stands for "human readable"
savepath = path
salt = r"$Li_2S_6$"

runs_inds = range(6, 11)
MDfiles = [f"HQ.{i}" for i in runs_inds]
runs = [f"{t*1000:.0f}_ps" for t in runs_inds]

# path = "/home/santi/MD/MDRelax_results/TEGDME/"
# savepath=path
# runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
# solvent = "TEGDME"

# print("WARNING!! revisar el dt, esta multiplicado por 5")


# Number of time steps
Ntimes = 10001
# Number of runs
Nruns = len(runs)
# Number of Li ions in a run 
Ncations = 4

efg_sources = [cation, anion, solvent_hr]
acf_cation = np.zeros([Ntimes, Nruns, Ncations])
acf_anion = np.zeros([Ntimes, Nruns, Ncations])
acf_solvent = np.zeros([Ntimes, Nruns, Ncations])
efg_variance = np.zeros([Nruns, Ncations])
acf_means = np.zeros([Ntimes, Nruns])
efg_variance_mean_over_cations=np.zeros([Nruns])
run_ind = -1
# Loop sobre runs para calcular ACF
for run in runs:
    print(f"RUN: {run} ==================")
    run_ind += 1    
    efg_cation_variance = np.zeros(Ncations)
    efg_anion_variance = np.zeros(Ncations)
    efg_solvent_variance = np.zeros(Ncations)    
    for efg_source in efg_sources: 
        filename = f"{path}/EFG_{efg_source}_{run}.dat"    
        data = np.loadtxt(filename)[:Ntimes, :]        
        # data columns order:    
        # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
        # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
        t0 = time.time()
        for nn in range(Ncations): #uno para cada litio
            # plt.plot(data[:,0], data[:,nn+1])
            t = data[:,0]
            Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
            Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                            
            EFG = np.array([[Vxx, Vxy, Vxz],
                            [Vxy, Vyy, Vyz],
                            [Vxz, Vyz, Vzz]])
            # Calculo ACF
            Ntau = t.size                               
            tau = np.zeros([Ntau])
            acf = np.zeros([Ntau])        
            efg_squared = np.zeros([Ntau])        
            dt = t[1]-t[0]
            for jj in range(Ntau):#--------------------    
                tau_jj = jj*dt            
                max_tau_index = t.size-jj            
                # jj, t0, acf_ii = 0, 0, 0            
                # while t0+tau<=times[-1]:
                acf_jj = 0                    
                for ii in range(0,max_tau_index):                 
                    # product = 0
                    # product += Vxx[ii]*Vxx[ii+jj] 
                    # product += Vyy[ii]*Vyy[ii+jj] 
                    # product += Vzz[ii]*Vzz[ii+jj]
                    # product += 2 * Vxy[ii]*Vxy[ii+jj]
                    # product += 2 * Vyz[ii]*Vyz[ii+jj]
                    # product += 2 * Vxz[ii]*Vxz[ii+jj]  
                    # acf_jj += product
                    product = EFG[:,:,ii]*EFG[:,:,ii+jj]             
                    acf_jj += np.sum(product)
                tau[jj] = tau_jj
                promedio = acf_jj/(max_tau_index+1)
                acf[jj] = promedio                                                           
                if jj%1000==0:                     
                    print(f"   {cation} {nn+1}/{Ncations}:   t = {t[jj]:.2f} ps")
                elif jj==Ntau-1:                     
                    print(f"   {cation} {nn+1}/{Ncations}:  ready")
            #-------------------------------------------
            # efg_squared = Vxx**2+Vyy**2+Vzz**2 + 2*(Vxy**2+Vyz**2+Vxz**2)                
            efg_squared = np.sum(EFG*EFG, axis=(0,1))
            if cation in efg_source:
                acf_cation[:, run_ind, nn] = acf
                efg_cation_variance[nn] = np.mean(efg_squared)                
            elif anion in efg_source:
                acf_anion[:, run_ind, nn] = acf
                efg_anion_variance[nn] = np.mean(efg_squared)
            elif solvent_hr in efg_source:
                acf_solvent[:, run_ind, nn] = acf
                efg_solvent_variance[nn] = np.mean(efg_squared)   
            tn = time.time()
            print(f"tiempo=   {tn-t0} s") 
    efg_variance[run_ind, :] = efg_cation_variance+\
                               efg_anion_variance +\
                               efg_solvent_variance

    #===================================================================    
    fig, ax = plt.subplots(num=run_ind+1)
    fig_cation, ax_cation = plt.subplots(num=(run_ind+1)*10)
    fig_anion, ax_anion = plt.subplots(num=(run_ind+1)*100)
    fig_solvent, ax_solvent = plt.subplots(num=(run_ind+1)*1000)
    
    for nn in range(Ncations):
        ax_cation.plot(tau, acf_cation[:,run_ind, nn],
                       label=f"{cation}{nn+1}", lw=2, alpha=0.5)
        ax_anion.plot(tau, acf_anion[:,run_ind, nn],
                      label=f"{cation}{nn+1}", lw=2, alpha=0.5)
        ax_solvent.plot(tau, acf_solvent[:,run_ind, nn],
                        label=f"{cation}{nn+1}", lw=2, alpha=0.5)        

    # promedio sobre cationes:
    acf_cation_promedio = np.mean(acf_anion[:,run_ind, :], axis=1)
    acf_anion_promedio = np.mean(acf_anion[:,run_ind, :], axis=1)
    acf_solvent_promedio = np.mean(acf_solvent[:,run_ind, :], axis=1)
    acf_means[:, run_ind] = acf_cation_promedio+\
                            acf_anion_promedio+\
                            acf_solvent_promedio
    efg_variance_mean_over_cations[run_ind] = np.mean(efg_variance[run_ind, :])                            

    # plot means in each graph
    ax_cation.plot(tau, acf_cation_promedio, label=f"Mean", 
                   lw=3, color='red')
    ax_anion.plot(tau, acf_anion_promedio, label=f"Mean", 
                  lw=3, color='blue')    
    ax_solvent.plot(tau, acf_solvent_promedio, label=f"Mean", 
                    lw=3, color='grey')
    # plot all efg-sources in a single graph
    ax.plot(tau, acf_means[:, run_ind],
            color='k', lw=3, label='Total ACF')
    ax.plot(tau, acf_cation_promedio, color='red',
            lw=2, label=f'EFG-source: {cation}')    
    ax.plot(tau, acf_anion_promedio, color='blue',
            lw=2, label=f'EFG-source: {anion}')
    ax.plot(tau, acf_solvent_promedio, color='grey',
            lw=2, label=f'EFG-source: {solvent_hr}')
    for ax_i, fig_i, source in zip([ax, ax_cation, ax_anion, ax_solvent],
                                   [fig, fig_cation, fig_anion, fig_solvent],
                                   ["total",cation, anion, solvent_hr]):            
        ax_i.axhline(0, color='k', ls='--')
        ax_i.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
        ax_i.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
        ax_i.legend()
        fig_i.suptitle(fr"{solvent_hr}$-Li_2S_6$ EFG Autocorrelation Function."+"\n"+\
                        f"EFG source: {source}", fontsize=16)
        fig_i.tight_layout()
        fig_i.savefig(f"{savepath}/Figuras/ACF_{run}_{source}.png")

    # guardo autocorrelaciones promedio
    data = np.array([tau, 
                    acf_anion_promedio+acf_solvent_promedio,
                    acf_anion_promedio, acf_solvent_promedio]).T
    header = "tau\tACF_total\tACF_sulfur\tACF_solvent\n"\
             "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/ACF_{run}.dat", data, header=header)
    
    # guardo varianzas promedio    
    data = np.array([efg_variance_mean_over_cations[run_ind]])
    header = f"EFG variance: mean over {Ncations} Li ions.\t"\
              "Units: e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/EFG_variance_{run}.dat", data, header=header)


    #FIGURA: ACF cumulativos:  -----------------------
    fig, ax = plt.subplots(num=(run_ind+1)*1000)
    
    data = acf_cation_promedio
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_cations[run_ind]
    ax.plot(tau, cumulative, label=f'EFG-source: {cation}',
            lw=2, color="red")
    
    data = acf_anion_promedio
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_cations[run_ind]
    ax.plot(tau, cumulative, label=f'EFG-source: {anion}',
            lw=2, color="blue")

    data = acf_solvent_promedio
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_cations[run_ind]
    ax.plot(tau, cumulative, label=f'EFG-source: {solvent_hr}',
            lw=2, color="grey")                                   
    
    # Primero promedio y luego integro:    
    integral = cumulative_simpson(acf_means[:, run_ind], x=tau, initial=0)
    cumulative_promedio = integral/efg_variance_mean_over_cations[run_ind]
    # grafico    
    ax.plot(tau, cumulative_promedio, label="Cumulative of ACF mean", lw=4, color='k')         
    ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")    
    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    title = f"{solvent_hr}-"\
            f"{salt}"+"\n"\
            r" Cumulative Integral of ACF:   "\
            r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
            r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
            r"$V_{\alpha\beta}(t')\rangle dt'$"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    corr_time_range = np.array([0.4, 0.8])*cumulative_promedio.size
    corr_time = np.mean(cumulative_promedio[int(corr_time_range[0]):int(corr_time_range[0])])
    ax.hlines(corr_time, 
             tau[int(corr_time_range[0])],
             tau[int(corr_time_range[1])], 
             ls='--', color='grey', lw = 1.5,
             label=f"~{corr_time:.1f} ps")

    ax.legend()
    fig.savefig(f"{savepath}/Figuras/CorrelationTime_{run}.png")
#%% Finally, the mean of all runs:
# FIGURA: Autocorrelaciones    
fig, ax = plt.subplots(num=3781781746813134613543546)
run_ind = -1
for run in runs:    
    run_ind += 1    
    ax.plot(tau, acf_means[:, run_ind], label=f"run: {runs[run_ind]}", 
            lw=2, color='grey', alpha=0.5)
# compute the mean over runs:            
acf_mean = np.mean(acf_means, axis=1)
ax.plot(tau, acf_mean, label=f"Mean over runs", lw=3, color='k')

efg_variance_mean_over_cations[run_ind] = np.mean(efg_variance[run_ind, :])                            

ax.axhline(0, color='k', ls='--')
ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
ax.legend()
fig.suptitle(fr"{solvent_hr}-{salt} EFG Autocorrelation Function", fontsize=16)
fig.tight_layout()
fig.savefig(f"{savepath}/Figuras/ACF_mean-over-runs.png")


# FIGURA: Cumulatives---------------------------------------------------
fig, ax = plt.subplots(num=37817817174681374681354132541354)
run_ind = -1
for run in runs:    
    run_ind += 1        
    data = acf_means[:, run_ind]
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_cations[run_ind]
    ax.plot(tau, cumulative, label=f"run: {runs[run_ind]}",
            lw=2, color="grey", alpha=0.5)                                   
    
# compute the mean over runs:            
######## ACA NO SE SI POMEDIAR LA VARIANZA ANTES O DESPUES DE INTEGRAR
data = acf_mean # mean over runs
efg_variance_mean_over_runs = np.mean(efg_variance_mean_over_cations[run_ind])
integral = cumulative_simpson(data, x=tau, initial=0)
cumulative = integral/efg_variance_mean_over_runs
ax.plot(tau, cumulative, label="Mean over runs", lw=3, color="k") 

ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
title = f"{solvent_hr}-"\
        f"{salt}"+"\n"\
        r" Cumulative Integral of ACF:   "\
        r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
        r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
        r"$V_{\alpha\beta}(t')\rangle dt'$"
fig.suptitle(title, fontsize=12)
fig.tight_layout()

corr_time_range = np.array([0.4, 0.8])*cumulative.size
corr_time = np.mean(cumulative[int(corr_time_range[0]):int(corr_time_range[0])])
ax.hlines(corr_time, 
         tau[int(corr_time_range[0])],
         tau[int(corr_time_range[1])], 
         ls='--', color='grey', lw = 1.5,
         label=f"~{corr_time:.1f} ps")

ax.legend()
fig.savefig(f"{savepath}/Figuras/CorrelationTime_mean-over-runs.png")


# %%
