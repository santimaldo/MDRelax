#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:00:58 2023

@author: santi

read ACF functions and average
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid, simpson
from pathlib import Path
import time
from pathlib import Path
from Functions import *
plt.rcParams.update({'font.size': 12})
    
# # DME - Li2S6
# path_MDrelax = "/home/santi/MD/MDRelax_results/DME_PS/"
# savepath = path_MDrelax
# cation_itp, anion_itp, solvent_itp = ["Li","S6", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","S6", "DME"] # names
# salt = r"Li$_2$S$_6$"
# runs_inds = range(6,11)
# mdp_file = "HQ"
# runs = [f"{t*1000:.0f}_ps" for t in runs_inds]
# # Number of Li ions in a run 
# Ncations = 4


# DME - LiTFSI
# path_MDrelax = "/home/santi/MD/MDRelax_results/DME_LiTFSI/"
# savepath = path_MDrelax + "test/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_DME", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "DME"] # names
# salt = "LiTFSI"
# runs_inds = range(6,11)
# mdp_file = "HQ"
# runs = [f"{t*1000:.0f}_ps" for t in runs_inds]
# # Number of Li ions in a run 
# Ncations = 4


# DME - No anion
path_MDrelax = "/home/santi/MD/MDRelax_results/DME_no-anion/"
savepath = path_MDrelax
cation_itp, anion_itp, solvent_itp = ["Li","none", "DME_7CB8A2"] # as in .itp files
cation, anion, solvent = ["Li","none", "DME"] # names
salt = r"Li$^+$"
runs_inds = range(6,11)
runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
runs_prefix = "HQ"

# Number of Li ions in a run 
Ncations = 1

def calculate_ACF(path_MDrelax,
                  savepath = None,
                  species = ["cation", "anion", "solvent"],                  
                  Ncations = 1,
                  runs_prefix = "HQ",
                  runs_suffix = None,                                    
                  ):
    """
    Reads EFG files and calculate ACF functions, plot the results
    and export plots
    """
    print("="*33)
    print("="*10+f"calculate_ACF"+"="*10)
    print("="*33)
    if savepath is None:
        savepath = path_MDrelax 
    # unpack species:
    cation, anion, solvent = species
    # creates the savepath if it doesn't exist
    Path(f"{savepath}/Figures").mkdir(parents=True, exist_ok=True)
    runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]
    # Number of time steps
    Ntimes = get_Ntimes(f"{path_MDrelax}EFG_{cation}_{runs[0]}.dat")    
    # Number of runs
    Nruns = len(runs)
    
    
    t0 = time.time()
    efg_sources = [cation, anion, solvent, "total"]
    EFG_cation = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_anion = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_solvent = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_total = np.zeros([3,3, Ntimes, Nruns, Ncations])
    acf_cation = np.zeros([Ntimes, Nruns, Ncations])
    acf_anion = np.zeros([Ntimes, Nruns, Ncations])
    acf_solvent = np.zeros([Ntimes, Nruns, Ncations])
    acf_total = np.zeros([Ntimes, Nruns, Ncations])
    efg_variance = np.zeros([Nruns, Ncations])
    acf_total_mean = np.zeros([Ntimes, Nruns])
    run_ind = -1
    print("Reading files...")
    # Loop sobre runs para calcular ACF
    for run in runs:
        print(f"RUN: {run} "+"="*30)
        run_ind += 1    
        efg_source_index = -1
        for efg_source in efg_sources: 
            efg_source_index += 1                
            filename = f"{path_MDrelax}/EFG_{efg_source}_{run}.dat"
            print("reading ", filename)
            data = np.loadtxt(filename)[:Ntimes, :]        
            # read the time only once:
            if (run_ind==0) and (efg_source_index==0):
                # tau and t are the same
                tau = data[:,0]            
            # data columns order:    
            # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
            # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
            t0 = time.time()
            for nn in range(Ncations): #uno para cada litio
                # plt.plot(data[:,0], data[:,nn+1])            
                Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
                Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                            
                # this EFG_nn is (3,3,Ntimes) shaped
                # nn is the cation index
                EFG_nn = np.array([[Vxx, Vxy, Vxz],
                                    [Vxy, Vyy, Vyz],
                                    [Vxz, Vyz, Vzz]])                                
                # EFG_{source} shape: (3,3, Ntimes, Nruns, Ncations)
                if cation in efg_source:            
                    EFG_cation[:,:,:, run_ind, nn] = EFG_nn            
                elif anion in efg_source:            
                    EFG_anion[:,:,:, run_ind, nn] = EFG_nn            
                elif solvent in efg_source:
                    EFG_solvent[:,:,:, run_ind, nn] = EFG_nn
                elif solvent in efg_source:
                    EFG_solvent[:,:,:, run_ind, nn] = EFG_nn 
                
    #Calculo ACF============================================== 
    #-------------------------------------------
    # calculating variance:
    efg_squared = np.sum(EFG_total*EFG_total, axis=(0,1))
    # efg_variance is the squared efg averaged over time:
    # shape: (Nruns, Ncations),
    efg_variance = np.mean(efg_squared, axis=0)
    #-------------------------------------------
    print("Calculating ACF...")        

    # calculate total ACF:
    print(rf"ACF with EFG_total")    
    acf_total = Autocorrelate(tau, EFG_total)

    # calculate only if a cation is a possible efg source
    print(rf"ACF with EFG-source: {cation}")
    if Ncations>1:     
        acf_cation = Autocorrelate(tau, EFG_cation)
    else:
        print(rf"There is only one {cation}. It can't be an EFG source")

    # calculate only if anions exist
    print(rf"ACF with EFG-source: {anion}")
    if "none" in anion.lower():
        print("since no anion is present, this step is skipped")
    else:    
        acf_anion = Autocorrelate(tau, EFG_anion)

    print(rf"ACF with EFG-source: {solvent}")
    acf_solvent = Autocorrelate(tau, EFG_solvent)
    tn = time.time()
    print(f"tiempo=   {tn-t0} s") 

    #-------------------------------------------
    # calculating means:

    #===================================================================    
    #FIGURA: ACF:  -----------------------
    # Mean values over carions:
    #
    # efg_variance_mean_over_cations.shape = [Ntimes]
    efg_variance_mean_over_cations = np.mean(efg_variance, axis=1)
        
    acf_total_mean = np.mean(acf_total, axis=2)
    acf_cation_mean = np.mean(acf_cation, axis=2)
    acf_anion_mean = np.mean(acf_anion, axis=2)
    acf_solvent_mean = np.mean(acf_solvent, axis=2)
    
    # calculating cross product terms of ACF
    acf_cross = acf_total - (acf_cation+acf_anion+acf_solvent)
    acf_cross_mean = np.mean(acf_cross, axis=2)

    # Saving Data =======================================
    # 0) EFG variance:  mean over runs and cations
    # 1) ACF:           mean over runs and cations
    # 2) EFG variance:  mean over cations, one per run
    # 3) ACF:           mean over cations, one per run
    # 4) ACF:           one per run, all cations data

    # 0)-------------------------------------------------
    ## save efg_variance_mean_over_runs data
    header = f"EFG variance: mean over runs.\t"\
              "Units: e^2*A^-6*(4pi*epsilon0)^-2"    
    np.savetxt(f"{savepath}/EFG_variance_mean-over-runs.dat", [efg_variance_mean_over_runs], header=header)
    
    # 1)-------------------------------------------------
    ## save acf_mean_over_runs data
    data = np.array([tau, 
                     np.mean(acf_total_mean, axis=1),
                     np.mean(acf_cation_mean, axis=1),
                     np.mean(acf_anion_mean, axis=1),
                     np.mean(acf_solvent_mean, axis=1),
                     np.mean(acf_cross_mean, axis=1)]).T
    header = f"tau\tACF_total\tACF_{cation}\tACF_{anion}"\
             f"\tACF_{solvent}\tACF_cross-terms\n"\
              "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/ACF_mean-over-runs.dat", data, header=header)

    
    run_ind = -1
    for run in runs:    
        run_ind += 1  
        # 2)-----------------------------------------------
        # guardo varianzas promedio    
        data = np.array([efg_variance_mean_over_cations[run_ind]])
        header = f"EFG variance: mean over {Ncations} Li ions.\t"\
                "Units: e^2*A^-6*(4pi*epsilon0)^-2"
        np.savetxt(f"{savepath}/EFG_variance_{run}.dat", data, header=header)        
        # 3)-----------------------------------------------
        # guardo autocorrelaciones promedio
        data = np.array([tau, 
                        acf_total_mean[:, run_ind],
                        acf_cation_mean[:, run_ind],
                        acf_anion_mean[:, run_ind], 
                        acf_solvent_mean[:, run_ind],
                        acf_cross_mean[:, run_ind]]).T
        header = f"tau\tACF_total\tACF_{cation}\tACF_{anion}"\
                 f"\tACF_{solvent}\tACF_cross-terms\n"\
                  "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
        np.savetxt(f"{savepath}/ACF_{run}.dat", data, header=header)
        # 4)-----------------------------------------------
        for acf, source in zip([acf_total, acf_cation,
                            acf_anion, acf_solvent],
                           ["total", cation, anion, solvent]):                   
            data += [acf[:, run_ind, nn] for nn in range(Ncations)] 
            data = np.array([tau]+ data).T
            header = f"tau\tACF_total for each cation\n"\
                    "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
            np.savetxt(f"{savepath}/ACF_efgsource-{source}_{run}_all-cation.dat", data, header=header)
    return 0




















def plotACF(path_MDrelax,
            savepath = None,
            species = ["cation", "anion", "solvent"],                  
            Ncations = 1,
            runs_prefix = "HQ",
            runs_suffix = None,                                                     
            salt = r"Li$_2$S$_6$"):
    """
    Read acf data and plot 
    """    
    # unpack species:
    cation, anion, solvent = species
    # define savepath
    if savepath is None:
        savepath = path_MDrelax 
    runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]
    # Number of time steps
    Ntimes = get_Ntimes(f"{path_MDrelax}EFG_{cation}_{runs[0]}.dat")    
    # Number of runs
    Nruns = len(runs)

    #=======================================================
    #====================READ===============================
    #=======================================================
    # 0)-------------------------------------------------
    ## READ efg_variance_mean_over_runs data    
    efg_variance_mean_over_runs = np.loadtxt(f"{path_MDrelax}/EFG_variance_mean-over-runs.dat")
    
    # 1)-------------------------------------------------
    ## read acf_mean_over_runs data    
    tau, acf_mean = np.loadtxt(f"{path_MDrelax}/ACF_mean-over-runs.dat")
    
    acf_cation = np.zeros([Ntimes, Nruns, Ncations])
    acf_anion = np.zeros([Ntimes, Nruns, Ncations])
    acf_solvent = np.zeros([Ntimes, Nruns, Ncations])
    acf_total = np.zeros([Ntimes, Nruns, Ncations])
    efg_variance = np.zeros([Nruns, Ncations])

    run_ind = -1
    for run in runs:    
        run_ind += 1  
        # 2)-----------------------------------------------
        # guardo varianzas promedio            
        efg_variance[run_ind, :] = np.loadtxt(f"{path_MDrelax}/EFG_variance_{run}.dat")[:,1:]
        # 3)-----------------------------------------------
        # guardo autocorrelaciones promedio        
        # np.loadtxt(f"{path_MDrelax}/ACF_{run}.dat", data, header=header)
        # 4)-----------------------------------------------
        acf_total[:,run_ind, :] = np.loadtxt(f"{path_MDrelax}/ACF_efgsource-total_{run}_all-cation.dat")[:,1:]
        acf_cation[:,run_ind, :] = np.loadtxt(f"{path_MDrelax}/ACF_efgsource-cation_{run}_all-cation.dat")[:,1:]
        acf_anion[:,run_ind, :] = np.loadtxt(f"{path_MDrelax}/ACF_efgsource-anion_{run}_all-cation.dat")[:,1:]
        acf_solvent[:,run_ind, :] = np.loadtxt(f"{path_MDrelax}/ACF_efgsource-solvent_{run}_all-cation.dat")[:,1:]  

    #------------------------------------------------------
    #------------------------------------------------------
    #------------------------------------------------------
    # Calculte
    acf_cross = acf_total - (acf_cation+acf_anion+acf_solvent)

    acf_total_mean = np.mean(acf_total, axis=2)
    acf_cation_mean = np.mean(acf_cation, axis=2)
    acf_anion_mean = np.mean(acf_anion, axis=2)
    acf_solvent_mean = np.mean(acf_solvent, axis=2)
    acf_cross_mean = np.mean(acf_cross, axis=2)


    #=======================================================
    #====================PLOTS==============================
    #=======================================================
    run_ind = -1
    for run in runs:    
        run_ind += 1   
        # Create a figure with 1 row and 3 columns for cation, anion, solvent
        fig_subplots, ax_subplots = plt.subplots(2, 2, figsize=(10, 10), num=(run_ind+1)*10)    
        # Create the separate figure for the mean ACF plot
        fig_mean, ax_mean = plt.subplots(num=(run_ind+1)*100, figsize=(8, 6))    
        # Assign the axes for the subplots
        ax_cation, ax_anion, ax_solvent, ax_cross = ax_subplots.flatten()
        
        # Plot cation data
        for nn in range(Ncations):
            ax_cation.plot(tau, acf_cation[:,run_ind, nn],
                        label=f"{cation}{nn+1}", lw=2, alpha=0.5)    
        # Plot anion data
        for nn in range(Ncations):
            ax_anion.plot(tau, acf_anion[:,run_ind, nn],
                        label=f"{cation}{nn+1}", lw=2, alpha=0.5)    
        # Plot solvent data
        for nn in range(Ncations):
            ax_solvent.plot(tau, acf_solvent[:,run_ind, nn],
                            label=f"{cation}{nn+1}", lw=2, alpha=0.5)
        # Plot cross-terms data
        for nn in range(Ncations):
            ax_cross.plot(tau, acf_cross[:,run_ind, nn],
                            label=f"{cation}{nn+1}", lw=2, alpha=0.5)                                    
        
        # Plot means in each graph
        ax_cation.plot(tau, acf_cation_mean[:,run_ind], label="Mean", lw=3, color='red')
        ax_anion.plot(tau, acf_anion_mean[:,run_ind], label="Mean", lw=3, color='blue')    
        ax_solvent.plot(tau, acf_solvent_mean[:,run_ind], label="Mean", lw=3, color='dimgrey')
        ax_cross.plot(tau, acf_cross_mean[:,run_ind], label="Mean", lw=3, color='orange')
                
        # Plot all efg-sources in a single graph
        ax_mean.plot(tau, acf_total_mean[:, run_ind], color='k', lw=3, label='Total ACF')
        ax_mean.plot(tau, acf_cross_mean[:, run_ind], color='orange', lw=2, label=f'EFG-source: {solvent}')
        ax_mean.plot(tau, acf_cation_mean[:, run_ind], color='red', lw=2, label=f'EFG-source: {cation}')    
        ax_mean.plot(tau, acf_anion_mean[:, run_ind], color='blue', lw=2, label=f'EFG-source: {anion}')
        ax_mean.plot(tau, acf_solvent_mean[:, run_ind], color='grey', lw=2, label=f'EFG-source: {solvent}')
        
        
        # Customize the subplots (cation, anion, solvent)
        titles = [f"EFG-source: {s}" for s in [cation, anion, solvent]]
        titles += ["Cross-terms of ACF"]
        # colors = ['red', 'blue', 'dimgrey', 'orange']
        for ax_i, title in zip(ax_subplots, titles):
            ax_i.axhline(0, color='k', ls='--')
            ax_i.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=14)
            ax_i.set_xlabel(r"$\tau$ [ps]", fontsize=14)    
            ax_i.set_title(title)
            ax_i.legend()
            
        title = f"{solvent}-{salt} : run: {run}.    "\
                r"$ACF(\tau) = $"\
                r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
                r"$V_{\alpha\beta}(\tau)\rangle$"    
        fig_subplots.suptitle(title, fontsize=16)
        fig_subplots.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        # Customize and save the mean figure
        ax_mean.axhline(0, color='k', ls='--')
        ax_mean.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=14)
        ax_mean.set_xlabel(r"$\tau$ [ps]", fontsize=14)
        ax_mean.legend()    
        fig_mean.suptitle(title, fontsize=16)
        fig_mean.tight_layout()    

        fig_subplots.savefig(f"{savepath}/Figures/ACF_bySource_{run}.png")
        fig_mean.savefig(f"{savepath}/Figures/ACF_{run}.png")

        #FIGURA: ACF cumulativos==============================
        fig, ax = plt.subplots(num=(run_ind+1)*10000)
        
        data = acf_cation_mean[:,run_ind]
        integral = cumulative_simpson(data, x=tau, initial=0)    
        cumulative = integral/efg_variance_mean_over_cations[run_ind]
        ax.plot(tau, cumulative, label=f'EFG-source: {cation}',
                lw=2, color="red")
        
        data = acf_anion_mean[:,run_ind]
        integral = cumulative_simpson(data, x=tau, initial=0)    
        cumulative = integral/efg_variance_mean_over_cations[run_ind]
        ax.plot(tau, cumulative, label=f'EFG-source: {anion}',
                lw=2, color="blue")

        data = acf_solvent_mean[:,run_ind]
        integral = cumulative_simpson(data, x=tau, initial=0)    
        cumulative = integral/efg_variance_mean_over_cations[run_ind]
        ax.plot(tau, cumulative, label=f'EFG-source: {solvent}',
                lw=2, color="grey")                                   
        
        # Primero promedio y luego integro:    
        data = acf_total_mean[:, run_ind]
        integral = cumulative_simpson(data, x=tau, initial=0)
        cumulative_promedio = integral/efg_variance_mean_over_cations[run_ind]
        # grafico    
        ax.plot(tau, cumulative_promedio, label="Cumulative of ACF mean", lw=4, color='k')         
        ax.legend(title=f"RUN {run_ind}", fontsize=10, loc="upper right")    
        ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
        ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
        title = f"{solvent}-"\
                f"{salt} :   run: {run}"+"\n"\
                r" Cumulative Integral of ACF:   "\
                r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
                r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
                r"$V_{\alpha\beta}(t')\rangle dt'$"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

        corr_time_range = np.array([0.4, 0.8])*cumulative_promedio.size
        corr_time = np.mean(cumulative_promedio[int(corr_time_range[0]):int(corr_time_range[1])])
        ax.hlines(corr_time, 
                tau[int(corr_time_range[0])],
                tau[int(corr_time_range[1])], 
                ls='--', color='grey', lw = 1.5,
                label=f"~{corr_time:.1f} ps")

        ax.legend()
        fig_subplots.savefig(f"{savepath}/Figures/CorrelationTime_bySource_{run}.png")
        fig_mean.savefig(f"{savepath}/Figures/CorrelationTime_{run}.png")
    #Finally, the mean of all runs:
    # FIGURA: Autocorrelaciones    
    fig, ax = plt.subplots(num=3781781746813134613543546)
    run_ind = -1
    for run in runs:    
        run_ind += 1    
        if run_ind == 0:  
            ax.plot(tau, acf_total_mean[:, run_ind], label="runs", 
                    lw=2, color='grey', alpha=0.5)
        else:  
            ax.plot(tau, acf_total_mean[:, run_ind], 
                    lw=2, color='grey', alpha=0.5)
                
    # compute the mean over runs:            
    acf_mean = np.mean(acf_total_mean, axis=1)
    ax.plot(tau, acf_mean, label=f"Mean over runs", lw=3, color='k')

    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    ax.legend()
    fig.suptitle(fr"{solvent}-{salt} EFG Autocorrelation Function", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{savepath}/Figures/ACF_mean-over-runs.png")

    # FIGURA: Cumulatives---------------------------------------------------
    fig, ax = plt.subplots(num=37817817174681374681354132541354)
    run_ind = -1
    for run in runs:    
        run_ind += 1        
        data = acf_total_mean[:, run_ind]
        integral = cumulative_simpson(data, x=tau, initial=0)    
        cumulative = integral/efg_variance_mean_over_cations[run_ind]
        if run_ind == 0:        
            ax.plot(tau, cumulative, label = "runs",
                lw=2, color="grey", alpha=0.5)
        else:
            ax.plot(tau, cumulative,
                lw=2, color="grey", alpha=0.5)
                                
        
    # compute the mean over runs:            
    ######## ACA NO SE SI POMEDIAR LA VARIANZA ANTES O DESPUES DE INTEGRAR
    data = acf_mean # mean over runs
    efg_variance_mean_over_runs = np.mean(efg_variance_mean_over_cations)
    integral = cumulative_simpson(data, x=tau, initial=0)
    cumulative = integral/efg_variance_mean_over_runs
    ax.plot(tau, cumulative, label="Mean over runs", lw=3, color="k") 

    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    title = f"{solvent}-"\
            f"{salt}"+"\n"\
            r" Cumulative Integral of ACF:   "\
            r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
            r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
            r"$V_{\alpha\beta}(t')\rangle dt'$"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    corr_time_range = np.array([0.4, 0.8])*cumulative.size
    corr_time = np.mean(cumulative[int(corr_time_range[0]):int(corr_time_range[1])])
    ax.hlines(corr_time, 
            tau[int(corr_time_range[0])],
            tau[int(corr_time_range[1])], 
            ls='--', color='grey', lw = 1.5,
            label=f"~{corr_time:.1f} ps")

    ax.legend()
    fig.savefig(f"{savepath}/Figures/CorrelationTime_mean-over-runs.png")

    
    return 0
