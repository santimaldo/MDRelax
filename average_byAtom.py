#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:00:58 2023

@author: santi

read ACF functions and average



COMPLETARRRRRRRRRRRR

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid, simpson
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
species_list = ["Li", "S6", "DME_7CB8A2"]

runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
solvent = "DME"

NPS = 2 # numero de polisulfuros
# Number of time steps
Ntimes = 10001
# Number of runs
Nruns = len(runs)
# Number of Li ions in a run 
NLi = 4

EFG_sources = ["sulfur", "solvent"]
acf_sulfur = np.zeros([Ntimes, Nruns, NLi])
acf_solvent = np.zeros([Ntimes, Nruns, NLi])
efg_variance = np.zeros([Nruns, NLi])
run_ind = -1
# Loop sobre runs para calcular ACF
for run in runs:
    run_ind += 1
    for efg_source in EFG_sources:
        acf_data = np.zeros([Ntimes, Nruns, NLi])
        filename = f"{path}/EFG_{efg_source}_{run}.dat"    
        data = np.loadtxt(filename)
        # data columns order:    
        # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
        # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
        for nn in range(NLi): #uno para cada litio
            # plt.plot(data[:,0], data[:,nn+1])
            t = data[:,0]
            Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
            Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                

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
        if "sulfur" in efg_source:
            acf_sulfur[:, run_ind, :] = acf_data[:, run_ind, :]
        elif "solvent" in efg_source:
            acf_solvent[:, run_ind, :] = acf_data[:, run_ind, :]        
#%% FIGURA: Autocorrelaciones
minimo = np.min([acf_sulfur, acf_solvent])
maximo = np.min([acf_sulfur, acf_solvent])

run_ind = -1
for run in runs:    
    run_ind += 1    
    fig, ax = plt.subplots(num=run_ind+1)
    fig_S, ax_S = plt.subplots(num=(run_ind+1)*10)
    fig_solvent, ax_solvent = plt.subplots((num=run_ind+1)*100)
    
    for nn in range(2*NPS): # 2, uno para cada litio                        
        ax_S.plot(tau, acf_sulfur[:,run_ind, nn],
                  label=f"Li{nn+1}", lw=2, alpha=0.5)
        ax_S.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])                      
        ax_solvent.plot(tau, acf_solvent[:,run_ind, nn],
                        label=f"Li{nn+1}", lw=2, alpha=0.5)
        ax_solvent.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])                      

    acf_sulfur_promedio = np.mean(acf_sulfur[:,run_ind, nn], axis=(1,2))
    acf_solvent_promedio = np.mean(acf_solvent[:,run_ind, nn], axis=(1,2))
    ax.plot(tau, acf_sulfur_promedio, color='gold',
            lw=2, label='EFG-source: sulfur')
    ax.plot(tau, acf_solvent_promedio, color='grey',
            lw=2, label='EFG-source: solvent')
    ax.plot(tau, acf_sulfur_promedio+acf_solvent_promedio,
            color='k', lw=3, label='Total ACF')

    for ax_i, source in zip([ax, ax_S, ax_solvent],
                            ["", "_Sulfur", "_solvent"]):            
    ax_i.axhline(0, color='k', ls='--')
    ax_i.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
    ax_i.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    ax_i.legend()
    fig_i.suptitle(fr"{solvent}$-Li_2S_6$ EFG{source} Autocorrelation Function", fontsize=22)
    fig_i.tight_layout()
    fig_i.savefig(f"{path}/Figuras/ACF_{run}{source}.png")


#%% FIGURA: ACF cumulativos:

### INCOMPLETO, CORREGIR

# fignum = 3
# run_ind = -1
# cumulative_data = np.zeros_like(acf_data)
# for run in runs:
#     run_ind += 1        
#     fig, ax = plt.subplots(num=(run_ind+1)*1000)
    
#     data = np.mean(acf_sulfur[:,run_ind, :], axis=(1,2))
#     integral = cumulative_simpson(data, x=tau, initial=0)    
#     cumulative = integral/efg_variance[run_ind, nn] 
#     ax.plot(tau, cumulative, label='EFG-source: sulfur',
#             lw=2, color="gold")               

#     data = np.mean(acf_solvent[:,run_ind, :], axis=(1,2))
#     integral = cumulative_simpson(data, x=tau, initial=0)    
#     cumulative = integral/efg_variance[run_ind, nn] 
#     ax.plot(tau, cumulative, label='EFG-source: sulfur',
#             lw=2, color="gold")                           
    
    
    
#     # Primero promedio y luego integro:
#     acf_promedio= 
#     integral = cumulative_simpson(acf_promedio, x=tau, initial=0)
#     cumulative_promedio = integral/np.mean(efg_variance)    
#     # grafico    
#     ax.plot(tau, cumulative_promedio, label="Cumulative of ACF mean", lw=4, color='k') 
        
#     ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")    
#     ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
#     ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
#     ax.yaxis.set_label_position("right")
#     ax.yaxis.tick_right()
#     title = f"{solvent}"\
#             r"$-Li_2S_6$"+"\n"\
#             r" Cumulative Integral of ACF:   "\
#             r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
#             r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
#             r"$V_{\alpha\beta}\rangle(t') dt'$"
#     fig.suptitle(title, fontsize=18)
#     fig.tight_layout()

#     corr_time = np.mean(cumulative_promedio[-int(0.4*cumulative_promedio.size):])
#     ax.axhline(corr_time, ls='--', color='grey', lw = 1.5,
#             label=f"correlation time: ~{corr_time:.1f} ps")
#     ax.legend()
#     fig.savefig(f"{path}/Figuras/CorrelationTime_{run}.png")

# # %%
