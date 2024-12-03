#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 2024

@author: santi

read MD data and calculate RMSD
"""
import time
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import matplotlib.pyplot as plt
import numpy as np


# path = "C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\Diffusion\Diglyme_PS/"
# savepath = "C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\Diffusion\Diglyme_PS/"
# file = "npt_diff"
# trajectory_extension = ".xtc"

path = "/home/santi/mendieta/Diglyme_small-boxes/Diglyme_LiTFSI/run_1ns/"
savepath = path
file = "NPT"
trajectory_extension = ".trr"

solvent = "DIG"
cation = "Lij"
anion = "TFS"
# defino para quien calculo:
selected = solvent


u = mda.Universe(path+file+".tpr", path+file+trajectory_extension)

# Selección de átomos
selection_str = f"resname {selected}"
selected_atoms = u.select_atoms(selection_str)
# Listas para almacenar los resultados
Ntimes = len(u.trajectory)
rmsd_results = np.zeros(Ntimes)
ti = time.time()                
# Iterar sobre los frames de referencia (t0)
np.arange(0,500001,5000)
for t0 in t0s: 
    u.trajectory[t0]  # Mover el universo al frame t0 (fijar referencia)

    # Configurar RMSD con el frame de referencia actual (t0)
    R = rms.RMSD(u,
                 select=selection_str,
                 ref_frame=t0  # Frame de referencia
                 )
    R.run()

    if t0==0:
        times = R.rmsd[:, 1]
        tf = time.time()
        print(f"tiempo de ejecucion (ref frame 0):   {tf-ti} s") 

    # Filtrar solo tiempos posteriores
    rmsd_post_t0 = R.rmsd[t0:, 2]  # RMSD (Ångstroms) para frames después de t0
    
    rmsd_results[0:len(rmsd_post_t0)] += rmsd_post_t0
    
# calculo el promedio de RMSD solo para el caso en que tienen la misma cantidad de frames    
last_time = Ntimes - t0s[-1]
rmsd = rmsd_results[:last_time]/len(t0s)
times = times[:last_time]
# Output de los resultados
tf = time.time()
print(f"tiempo de ejecucion total:   {tf-ti} s") 

# selection_str = f"resname {selected}"
# selected_atoms = u.select_atoms(selection_str)
# t0 = time.time()                
# R = rms.RMSD(u,
#              select=selection_str
#              )
# R.run()
# tf = time.time()
# print(f"tiempo de ejecucion:   {tf-t0} s") 

#%%
# rmsd = R.results.rmsd   # transpose makes it easier for plotting
# header = "index\ttime[ps]\tRMSD[Ang]\n"
# header += f"based on {selected_atoms.n_atoms} atoms from {selected} group"
# np.savetxt(savepath+file+f"_{selected}-RMSD.dat", rmsd, header=header)
# time = rmsd[:,1]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(times/1000, rmsd, label=selected)
ax.legend(loc="best")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"RMSD ($\AA$)")
fig.savefig(f"{savepath}RMSD_{selected}.png")
# %%
