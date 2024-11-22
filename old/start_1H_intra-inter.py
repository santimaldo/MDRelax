#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 20  2024

@author: santi

First script to test the 1H relaxation summation

vamos a iterar por los solventes, calculando el T1 de 1H total.
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rdf
import pandas as pd
import nmrformd as nmrmd
import time
from scipy.integrate import cumulative_trapezoid


# solvents = ["DOL","DME","Diglyme","TEGDME","ACN"]

solvents = ["ACN", "DME", "DOL","Diglyme","TEGDME",]
nmolecs = [200, 100, 200, 100, 100]







start_total_time = time.time()
T1s = []
T1s_intra = []
T1s_inter = []
n_sol = -1
for solvent, nmolec in zip(solvents, nmolecs):
    n_sol+=1
    # path = rf"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\{solvent}\run_1ns/"
    # savepath = fr"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\1H/"
    path = rf"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\CHARMM\{solvent}/nmolec_{nmolec}/"
    savepath = fr"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\CHARMM\solvents_AllProton/"
    print("="*50)
    print(f"solvente:\t{solvent}")
    print("Leyendo...")
    start_time = time.time()
    u = mda.Universe(path+"HQ.6.tpr", path+"HQ.6.xtc")
    elapsed_time = time.time() - start_time
    print(f"leido--- check v/   (en {elapsed_time:.2f} s)")


    # u.transfer_to_memory(stop=10001)
    dt = u.trajectory.dt # units of ps
    # ni = 1000 # "number_i"
    ni = 0 # AllH
    box = u.dimensions
    H_group = u.select_atoms("name H*")
        
    start_time = time.time()
    print("Calculando ACFS:")
    print("intramolecular...") #(si no es intra, es inter_molecular)
    nmr_H_intra = nmrmd.NMR(u, atom_group=H_group, isotropic=True,# actual_dt=dt, 
                                number_i=ni, type_analysis="intra_molecular", neighbor_group=H_group)
    elapsed_time = time.time() - start_time    
    print(f"\t elapsed: {elapsed_time/60:.2f} min")    
    
    print("intermolecular...") 
    start_time = time.time()
    nmr_H_inter = nmrmd.NMR(u, atom_group=H_group, isotropic=True,# actual_dt=dt, 
                                number_i=ni, type_analysis="inter_molecular", neighbor_group=H_group)
    elapsed_time = time.time() - start_time    
    print(f"\t elapsed: {elapsed_time/60:.2f} min")

    print("Graficando y guardando datos...")
    start_time = time.time()
    H_intra_T1 = nmr_H_intra.T1
    H_inter_T1 = nmr_H_inter.T1
    H_T1 = 1/(1/H_inter_T1+1/H_intra_T1)
    T1s.append(H_T1)
    T1s_intra.append(H_intra_T1)
    T1s_inter.append(H_inter_T1)    
    print(f"{solvent}-1H-T1_intra: {H_intra_T1:.2e} s")
    print(f"{solvent}-1H-T1_inter: {H_inter_T1:.2e} s")
    print(f'Time elapsed: {elapsed_time/60} minutes')
    
    ACF_intra = nmr_H_intra.gij[0,:]
    ACF_inter = nmr_H_inter.gij[0,:]
    
    tau = np.arange(ACF_intra.size)*dt
    #%%    
    data = np.array([tau, ACF_intra, ACF_inter]).T
    if ni!=0:
        header = f"tau (ps)    ACF ### ACF_intra   ACF_inter \n "\
                 f"calculated with {ni} atoms (over {H_group.n_atoms} total H atoms)"
    else:
        header = f"tau (ps)    ACF_intra   ACF_inter \n "\
                 f"calculated all H atoms: {H_group.n_atoms}"
    header += "\n"\
              f"T1intra = {H_intra_T1:.2e} s \n"\
              f"T1inter = {H_inter_T1:.2e} s \n"\
              f"T1total = {H_T1:.2e} s"
    np.savetxt(savepath+f"1H_ACF_{solvent}.dat", data, header=header)


    fig, ax = plt.subplots(num=n_sol)
    ax.set_title(f"{solvent}"+r" $^1H-^1H$ dipolar ACF")
    ax.plot(tau, ACF_intra.T, 'o-', label="H_intra")
    ax.plot(tau, ACF_inter.T, 'o-', label="H_inter")    
    ax.axhline(0, linestyle='--', color='k')    
    ax.legend()
    ax.set_xlabel(r"$\tau$ [ps]")
    ax.set_ylabel("ACF")
    fig.savefig(savepath+f"{solvent}_ACF.png")
    elapsed_time = time.time() - start_time   
    print(f"\t elapsed: {elapsed_time/60:.2f} min")


fig, ax = plt.subplots(num=67862786)
ax.bar(solvents, T1s)
fig.suptitle(r"Molecular Dynamics $^1H$ $T_1$")
fig.savefig(savepath+"T1.png")


print("TERMINADO!")
elapsed_time = time.time() - start_total_time   
print(f"\t tiempo total: {elapsed_time/3600:.2f} horas")