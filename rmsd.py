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

path = "C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\Diffusion\Diglyme_PS/"
savepath = "C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\Diffusion\Diglyme_PS/"
file = "npt_diff"

solvent = "DIG"
cation = "Lij"
anion = "TFS"

trajectory_extension = ".xtc"
# defino para quien calculo:
selected = cation


u = mda.Universe(path+file+".tpr", path+file+trajectory_extension)

selection_str = f"resname {selected}"
selected_atoms = u.select_atoms(selection_str)
t0 = time.time()                
R = rms.RMSD(u,
             select=selection_str
             )
R.run()
tf = time.time()
print(f"tiempo de ejecucion:   {tf-t0} s") 

#%%
rmsd = R.results.rmsd   # transpose makes it easier for plotting
header = "index\ttime[ps]\tRMSD[Ang]\n"
header += f"based on {selected_atoms.n_atoms} atoms from {selected} group"
np.savetxt(savepath+file+f"_{selected}-RMSD.dat", rmsd, header=header)

time = rmsd[:,1]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(time/1000, rmsd[:,2], label=selected)
ax.legend(loc="best")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"RMSD ($\AA$)")
fig.savefig(f"{savepath}RMSD_{selected}.png")
# %%
