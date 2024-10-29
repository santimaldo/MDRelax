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
runs = [f"{t:.1f}_ps" for t in [6000]]
solvent = "DME"


# Number of time steps
Ntimes = 10001
# Number of runs
Nruns = len(runs)
# Number of Li ions in a run 
NLi = 4


acf_data = np.zeros([Ntimes, Nruns, NLi])
efg_variance = np.zeros([Nruns, NLi])
run_ind = 0
# Loop sobre runs para calcular ACF
for run in runs:      
    filename = f"{path}/EFG_{run}.dat"    
    data = np.loadtxt(filename)
    # data columns order:    
    # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
    # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
    for nn in range(NLi): #uno para cada litio
        # plt.plot(data[:,0], data[:,nn+1])
        t = data[:,0]
        Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
        Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]
                
        plt.figure(1)
        plt.plot(t, np.array([Vxx, Vyy, Vzz, Vxy, Vyz, Vxz]).T)                
        plt.plot(t, Vxx+Vyy+Vzz, 'k', label='Trace')                
        plt.legend(["Vxx", "Vyy", "Vzz", "Vxy", "Vyz", "Vxz", "Trace: Vxx+Vyy+Vzz"],
                    ncols=4, fontsize=10)
        plt.title(f"{solvent};   run:{run};   Li{nn+1}")
        plt.xlabel("Time [ps]")
        plt.ylabel(r"EFG [$e/\AA^3$]")
        plt.tight_layout()
        plt.savefig(f"{path}Figuras/EFG_{run}_Li{nn+1}.png")
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
# fignum = 2
# fig = plt.figure(num=fignum, figsize=(12, 8))
# colors = ['k', 'b', 'r', 'g']
# run_ind = -1
# for frame_time in frame_times:
#   ii+=1
#   for run in runs:
#       jj+=1
#       run_ind += 1
#       jjcond = jj
#       if ii>0:
#           jjcond = jj-2
#       ax = plt.subplot2grid((int(Nruns/2),Nruns), (ii, jjcond))
#       ax.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])
#       if jj%2==1:
#           ax.yaxis.set_ticklabels([])
#       else:
#           ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
      
#       ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
#       for nn in range(2): # 2, uno para cada litio
#         acf = acf_data[:,run_ind, nn]
#         alpha = 1        
#         if nn>0: alpha=0.5
#         ax.plot(tau, acf, label=f"Li{nn+1}", lw=2,
#                  color=colors[run_ind], alpha=alpha)                    
#       ax.axhline(0, color='k', ls='--')
#       ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")

      
# ax = plt.subplot2grid((int(Nruns/2),Nruns), (0,int(Nruns/2)),
#                       rowspan=int(Nruns/2),colspan=int(Nruns/2))

# acf_promedio = np.mean(acf_data, axis=(1,2))
# datos = np.array([tau, acf_promedio]).T
# np.savetxt(f"{path}/ACF-mean.dat", datos)

# label = "Mean ACF"
# ax.plot(tau, acf_promedio, color='orange', label=label, lw=3)        
# ax.axhline(0, color='k', ls='--')
# ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
# ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
# ax.yaxis.set_label_position("right")
# ax.yaxis.tick_right()
# ax.legend()
# fig.suptitle(fr"{solvent}$-Li_2S_6$ EFG Autocorrelation Function", fontsize=22)
# fig.tight_layout()
# fig.savefig(f"{path}/Figuras/ACF.png")

# #%%
# # graficos:
# fignum = 3
# fig = plt.figure(num=fignum, figsize=(12, 8))
# colors = ['k', 'b', 'r', 'g']
# run_ind = -1
# ii, jj = -1,-1

# cumulative_data = np.zeros_like(acf_data)
# for frame_time in frame_times:
#   ii+=1
#   for run in runs:
#       jj+=1
#       run_ind += 1
#       jjcond = jj
#       if ii>0:
#           jjcond = jj-2
#       ax = plt.subplot2grid((int(Nruns/2),Nruns), (ii, jjcond))
#       # ax.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])
#       if jj%2==1:
#           ax.yaxis.set_ticklabels([])
#       else:
#           ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
      
#       ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
#       for nn in range(2): # 2, uno para cada litio
#         acf = acf_data[:,run_ind, nn]
#         alpha = 1        
#         if nn>0: alpha=0.5
        
#         # integral = cumulative_trapezoid(acf, x=tau, initial=0)
#         integral = cumulative_simpson(acf, x=tau, initial=0)
#         cumulative = integral/efg_variance[run_ind, nn] 
#         ax.plot(tau, cumulative, label=f"Li{nn+1}", lw=2,
#                  color=colors[run_ind], alpha=alpha)           
#         # guardo los datos para luego promediar
#         cumulative_data[:, run_ind, nn] = cumulative
#       ax.axhline(0, color='k', ls='--')
#       ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")      
      
      
# ax = plt.subplot2grid((int(Nruns/2),Nruns), (0,int(Nruns/2)),
#                       rowspan=int(Nruns/2),colspan=int(Nruns/2))     
# # Primero promedio y luego integro:
# acf_promedio= np.mean(acf_data, axis=(1,2))
# integral = cumulative_simpson(acf_promedio, x=tau, initial=0)
# cumulative_promedio = integral/np.mean(efg_variance)
# # guardo datos
# datos = np.array([tau, cumulative_promedio]).T
# np.savetxt(f"{path}/Cumulative-mean.dat", datos)
# # grafico
# label = "Cumulative of ACF mean"
# ax.plot(tau, cumulative_promedio, label=label, lw=3, color='orange') 

# # calculo el promedio de las cumulativas:
# # promedio_de_cumulatives = np.mean(cumulative_data, axis=(1,2))
# # label = "Mean of Cumulatives"
# # ax.plot(tau, promedio_de_cumulatives, '--', color='gray', label=label, lw=2) 

       
# ax.axhline(0, color='k', ls='--')
# ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
# ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
# ax.yaxis.set_label_position("right")
# ax.yaxis.tick_right()
# ax.legend()
# title = f"{solvent}"\
#         r"$-Li_2S_6$"+"\n"\
#         r" Cumulative Integral of ACF:   "\
#         r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
#         r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
#         r"$V_{\alpha\beta}\rangle(t') dt'$"
# fig.suptitle(title, fontsize=18)
# fig.tight_layout()
# fig.savefig(f"{path}/Figuras/CorrelationTime.png")

#%% FIGURA: Autocorrelaciones
minimo = np.min(acf_data)
maximo = np.min(acf_data)

colors = ['k', 'b', 'r', 'g']
run_ind = -1
for run in runs:    
    run_ind += 1    
    fig, ax = plt.subplots(num=run_ind+1, figsize=(8, 6))
    ax.set_ylim([1.1*np.min(acf_data), 1.1*np.max(acf_data)])    
    ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)    
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
    for nn in range(4): # 2, uno para cada litio
        acf = acf_data[:,run_ind, nn]
        alpha=0.5
        ax.plot(tau, acf, label=f"Li{nn+1}", lw=2,
                #color=colors[run_ind], 
                alpha=alpha)                        
    # ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")

    acf_promedio = np.mean(acf_data, axis=(1,2))
    # datos = np.array([tau, acf_promedio]).T
    # np.savetxt(f"{path}/ACF-mean.dat", datos)

    label = "Mean ACF"
    ax.plot(tau, acf_promedio, color='k', label=label, lw=3)        
    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yalim([-1.1*np.abs(minimo), 1.1*maximo])
    ax.legend()
    fig.suptitle(fr"{solvent}$-Li_2S_6$ EFG Autocorrelation Function", fontsize=22)
    fig.tight_layout()
    fig.savefig(f"{path}/Figuras/ACF_{run}.png")


#%% FIGURA: ACF cumulativos:
run_ind = -1
cumulative_data = np.zeros_like(acf_data)
for run in runs:
    run_ind += 1
    fig, ax = plt.subplots(num=(run_ind)*100, figsize=(8, 6))
    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)    
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
    for nn in range(4):
        acf = acf_data[:,run_ind, nn]
        alpha=0.5
    
        # integral = cumulative_trapezoid(acf, x=tau, initial=0)
        integral = cumulative_simpson(acf, x=tau, initial=0)
        cumulative = integral/efg_variance[run_ind, nn] 
        ax.plot(tau, cumulative, label=f"Li{nn+1}", lw=2,
                    #color=colors[run_ind], 
                    alpha=alpha)           
        # guardo los datos para luego promediar
        cumulative_data[:, run_ind, nn] = cumulative    
    ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")      
    # Primero promedio y luego integro:
    acf_promedio= np.mean(acf_data, axis=(1,2))
    integral = cumulative_simpson(acf_promedio, x=tau, initial=0)
    cumulative_promedio = integral/np.mean(efg_variance)
    # guardo datos
    # datos = np.array([tau, cumulative_promedio]).T
    # np.savetxt(f"{path}/OutputData/Cumulative-mean_{run}.dat", datos)
    # grafico
    label = "Cumulative of ACF mean"
    ax.plot(tau, cumulative_promedio, label=label, lw=4, color='k') 
    # calculo el promedio de las cumulativas:
    # promedio_de_cumulatives = np.mean(cumulative_data, axis=(1,2))
    # label = "Mean of Cumulatives"
    # ax.plot(tau, promedio_de_cumulatives, '--', color='gray', label=label, lw=2)      
    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    title = f"{solvent}"\
            r"$-Li_2S_6$"+"\n"\
            r" Cumulative Integral of ACF:   "\
            r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
            r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
            r"$V_{\alpha\beta}\rangle(t') dt'$"
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()

    corr_time = np.mean(cumulative_promedio[-int(0.4*cumulative_promedio.size):])
    ax.axhline(corr_time, ls='--', color='grey', lw = 1.5,
            label=f"correlation time: ~{corr_time:.1f} ps")
    ax.legend()
    fig.savefig(f"{path}/Figuras/CorrelationTime_{run}.png")

# %%
