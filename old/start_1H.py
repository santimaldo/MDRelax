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


solvents = ["DOL","DME","Diglyme","TEGDME","ACN"]
solvents = ["Diglyme","TEGDME"]

T1s = []
T2s = []
n_sol = -1
for solvent in solvents:
    n_sol+=1
    path = rf"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\{solvent}\run_1ns/"
    savepath = fr"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\1H/"
    print("Leyendo...")
    u = mda.Universe(path+"HQ.6.tpr", path+"HQ.6.xtc")
    print("listo")

    u.transfer_to_memory(stop=10001)
    dt = u.trajectory.dt # units of ps
    ni = 100 # "number_i"
    # ni = 0 # AllH
    box = u.dimensions

    H_group = u.select_atoms("name H*")
    # Li_group = u.select_atoms("name Li*")    
    H_free = H_group

    # # selecciono los atomos de H de la "primera esfera"------------------
    # #Define a distance threshold
    # distance_threshold = 3.8
    # # Compute the distance array
    # distances = mda.lib.distances.distance_array(H_group, Li_group)
    # # Create a mask for atoms within the distance threshold
    # within_threshold = distances < distance_threshold
    # # Determine which H atoms are within the threshold
    # H_group_within_threshold = H_group[np.any(within_threshold, axis=1)]
    # # Create a new AtomGroup called H_bond with the selected atoms
    # H_bond = u.select_atoms('index ' + ' '.join(map(str, H_group_within_threshold.indices)))
    # print("Number of selected H_bond atoms:", len(H_bond))
    # #---------------------------------------------------------------------
    # H_free = H_group.difference(H_bond)
    # n_i = len(H_bond)
    # start_time = time.time()
    # print("Comenzazmos con H_bond")
    # print("calculando intra...") #(si no es intra, es inter_molecular)
    # nmr_H_bond_intra = nmrmd.NMR(u, atom_group=H_bond, isotropic=True, #actual_dt=dt, 
    #                             type_analysis="intra_molecular", neighbor_group=H_group)
    # print("calculando inter...")
    # nmr_H_bond_inter= nmrmd.NMR(u, atom_group=H_bond, isotropic=True, #actual_dt=dt,
    #                          type_analysis="inter_molecular", neighbor_group=H_group)
    # print("calculando total...")
    # nmr_H_bond= nmrmd.NMR(u, atom_group=H_bond, isotropic=True, #actual_dt=dt
    #                         neighbor_group=H_group)
    # elapsed_time = time.time() - start_time
    # print(f'Time elapsed: {elapsed_time/60} minutes')
    #---------------------------------------------------------------
    start_time = time.time()
    print("Continuamos con H_free")
    # print("calculando intra...") #(si no es intra, es inter_molecular)
    # nmr_H_free_intra = nmrmd.NMR(u, atom_group=H_free, isotropic=True,# actual_dt=dt, 
    #                             number_i=ni, type_analysis="intra_molecular", neighbor_group=H_group)
    # print("calculando inter...")
    # nmr_H_free_inter = nmrmd.NMR(u, atom_group=H_free, isotropic=True,# actual_dt=dt, 
    #                             number_i=ni, type_analysis="inter_molecular", neighbor_group=H_group)
    print("calculando total...")
    nmr_H_free = nmrmd.NMR(u, atom_group=H_free, isotropic=True, #actual_dt=dt, 
                            number_i=ni, neighbor_group=H_group)
    elapsed_time = time.time() - start_time
    H_free_T1 = nmr_H_free.T1
    H_free_T2 = nmr_H_free.T2
    T1s.append(H_free_T1)
    T2s.append(H_free_T2)    
    print(f"{solvent}-1H-T1: {H_free_T1:.2e} s")
    print(f"{solvent}-1H-T2: {H_free_T2:.2e} s")
    print(f'Time elapsed: {elapsed_time/60} minutes')

    # ACF_bond_intra = nmr_H_bond_intra.gij[0,:]
    # ACF_bond_inter = nmr_H_bond_inter.gij[0,:]
    # ACF_bond = nmr_H_bond.gij[0,:]
    # ACF_free_intra = nmr_H_free_intra.gij[0,:]
    # ACF_free_inter = nmr_H_free_inter.gij[0,:]
    ACF_free = nmr_H_free.gij[0,:]
    #
    tau = np.arange(ACF_free.size)*dt

    # data = np.array([tau, ACF_free, ACF_intra, ACF_inter]).T
    data = np.array([tau, ACF_free]).T
    if ni!=0:
        header = f"tau (ps)    ACF \n "\
                 f"calculated with {ni} atoms (over {H_group.n_atoms} total H atoms)"
    else:
        header = f"tau (ps)    ACF \n "\
                 f"calculated all H atoms: {H_group.n_atoms}"
    header += "\n"\
              f"T1 = {H_free_T1:.2e} s \n"\
              f"T2 = {H_free_T2:.2e} s"
    np.savetxt(savepath+f"1H_ACF_{solvent}.dat", data, header=header)


    plt.figure(n_sol)
    plt.title(solvent)
    # plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_intra.T, 'o-', label="H_bond_intra")
    # plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_inter.T, 'o-', label="H_bond_inter")
    # plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond.T, 'o-', label="H_bond")
    # plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_intra.T, 'o-', label="H_free_intra")
    # plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_inter.T, 'o-', label="H_free_inter")
    plt.plot(tau, ACF_free, 'o-', label="H_free")
    plt.axhline(0, linestyle='--', color='k')
    plt.legend()
    plt.xlabel(r"$\tau$ [ps]")
    plt.ylabel("ACF")
    plt.savefig(savepath+f"{solvent}_ACF.png")

fig, ax = plt.subplots(num=67862786)
ax.bar(solvents, T1s)
fig.suptitle(r"Molecular Dynamics $^1H$ $T_1$")
fig.savefig(savepath+"T1.png")
