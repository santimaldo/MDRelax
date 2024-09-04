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

# path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/MDRelax/"
# species_list = ["Li", "S6", "DME_7CB8A2"]
# savepath = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/MDRelax/"
# runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
# solvent = "DME"

path = "/home/santi/MD/MDRelax_results/TEGDME/"
savepath=path
runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
solvent = "TEGDME"

print("WARNING!! revisar el dt, esta multiplicado por 5")


NPS = 2 # numero de polisulfuros
# Number of time steps
Ntimes = 50001
# Number of runs
Nruns = len(runs)
# Number of Li ions in a run 
NLi = 2*NPS


acf_sulfur = np.zeros([Ntimes, Nruns, NLi])
acf_solvent = np.zeros([Ntimes, Nruns, NLi])
efg_variance = np.zeros([Nruns, NLi])
run_ind = -1
# Loop sobre runs para calcular ACF
for run in runs:
    print(f"RUN: {run} ==================")
    run_ind += 1
    efg_sulfur_variance = np.zeros(NLi)
    efg_solvent_variance = np.zeros(NLi)
    for efg_source in ["sulfur", "solvent"]:        
        filename = f"{path}/EFG_{efg_source}_{run}.dat"    
        data = np.loadtxt(filename)[:Ntimes, :]        
        # data columns order:    
        # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
        # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
        t0 = time.time()
        for nn in range(NLi): #uno para cada litio
            # plt.plot(data[:,0], data[:,nn+1])
            t = data[:,0]*5
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
                    print(f"   Li {nn+1}/{NLi}:   t = {t[jj]:.2f} ps")
                elif jj==Ntau-1:                     
                    print(f"   Li {nn+1}/{NLi}:  ready")
            #-------------------------------------------
            # efg_squared = Vxx**2+Vyy**2+Vzz**2 + 2*(Vxy**2+Vyz**2+Vxz**2)                
            efg_squared = np.sum(EFG*EFG, axis=(0,1))
            if "sulfur" in efg_source:
                acf_sulfur[:, run_ind, nn] = acf
                efg_sulfur_variance[nn] = np.mean(efg_squared)
            elif "solvent" in efg_source:
                acf_solvent[:, run_ind, nn] = acf
                efg_solvent_variance[nn] = np.mean(efg_squared)   
            tn = time.time()
            print(f"tiempo=   {tn-t0} s") 
    efg_variance[run_ind, :] = efg_solvent_variance + efg_sulfur_variance

    #===================================================================
    # FIGURA: Autocorrelaciones    
    fig, ax = plt.subplots(num=run_ind+1)
    fig_S, ax_S = plt.subplots(num=(run_ind+1)*10)
    fig_solvent, ax_solvent = plt.subplots(num=(run_ind+1)*100)
    
    for nn in range(NLi): # 2, uno para cada litio                        
        ax_S.plot(tau, acf_sulfur[:,run_ind, nn],
                  label=f"Li{nn+1}", lw=2, alpha=0.5)                             
        ax_solvent.plot(tau, acf_solvent[:,run_ind, nn],
                        label=f"Li{nn+1}", lw=2, alpha=0.5)        

    acf_sulfur_promedio = np.mean(acf_sulfur[:,run_ind, :], axis=1)
    ax_S.plot(tau, acf_sulfur_promedio, label=f"Mean", 
              lw=3, color='gold')
    acf_solvent_promedio = np.mean(acf_solvent[:,run_ind, :], axis=1)
    ax_solvent.plot(tau, acf_solvent_promedio, label=f"Mean", 
                    lw=3, color='grey')
    ax.plot(tau, acf_sulfur_promedio+acf_solvent_promedio,
            color='k', lw=3, label='Total ACF')
    ax.plot(tau, acf_sulfur_promedio, color='gold',
            lw=2, label='EFG-source: sulfur')
    ax.plot(tau, acf_solvent_promedio, color='grey',
            lw=2, label='EFG-source: solvent')
    for ax_i, fig_i, source in zip([ax, ax_S, ax_solvent],
                                   [fig, fig_S, fig_solvent],
                                   ["", "_Sulfur", "_solvent"]):            
        ax_i.axhline(0, color='k', ls='--')
        ax_i.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
        ax_i.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
        ax_i.legend()
        fig_i.suptitle(fr"{solvent}$-Li_2S_6$ EFG{source} Autocorrelation Function", fontsize=16)
        fig_i.tight_layout()
        fig_i.savefig(f"{savepath}/Figuras/ACF_{run}{source}.png")

    # guardo autocorrelaciones promedio
    data = np.array([tau, 
                    acf_sulfur_promedio+acf_solvent_promedio,
                    acf_sulfur_promedio, acf_solvent_promedio]).T
    header = "tau\tACF_total\tACF_sulfur\tACF_solvent\n"\
             "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/ACF_{run}.dat", data, header=header)
    
    # guardo varianzas promedion
    efg_variance_mean_over_Li = np.mean(efg_variance[run_ind, :])
    data = np.array([efg_variance_mean_over_Li])
    header = f"EFG variance: mean over {NLi} Li ions.\t"\
              "Units: e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/EFG_variance_{run}.dat", data, header=header)


    #FIGURA: ACF cumulativos:  -----------------------
    fig, ax = plt.subplots(num=(run_ind+1)*1000)
    
    data = acf_sulfur_promedio
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_Li
    ax.plot(tau, cumulative, label='EFG-source: sulfur',
            lw=2, color="gold")               

    data = acf_solvent_promedio
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance_mean_over_Li
    ax.plot(tau, cumulative, label='EFG-source: solvent',
            lw=2, color="grey")                                   
    
    # Primero promedio y luego integro:
    acf_promedio= acf_sulfur_promedio + acf_solvent_promedio
    integral = cumulative_simpson(acf_promedio, x=tau, initial=0)
    cumulative_promedio = integral/efg_variance_mean_over_Li
    # grafico    
    ax.plot(tau, cumulative_promedio, label="Cumulative of ACF mean", lw=4, color='k')         
    ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")    
    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    title = f"{solvent}"\
            r"$-Li_2S_6$"+"\n"\
            r" Cumulative Integral of ACF:   "\
            r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
            r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
            r"$V_{\alpha\beta}\rangle(t') dt'$"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    corr_time = np.mean(cumulative_promedio[int(0.4*cumulative_promedio.size):int(0.9*cumulative_promedio.size)])
    ax.axhline(corr_time, ls='--', color='grey', lw = 1.5,
            label=f"correlation time: ~{corr_time:.1f} ps")
    ax.legend()
    fig.savefig(f"{savepath}/Figuras/CorrelationTime_{run}.png")

# %%
