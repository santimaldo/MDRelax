#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  23 12024

@author: santi

read .gro data and calculate dipole moment.
"""


import numpy as np
from Functions import *
import MDAnalysis as mda
import matplotlib.pyplot as plt


# path_Gromacs = "/home/santi/mendieta/TEGDME_LiTFSI/"
path_Gromacs = "/home/santi/mendieta/DME_small-boxes/DME/"
path_MDrelax = "/home/santi/MD/MDRelax_results/TEGDME_LiTFSI/"

get_Charges("S6", path_Gromacs)

u = mda.Universe(f"{path_Gromacs}HQ.6.tpr", f"{path_Gromacs}HQ.6.trr")

anions = u.select_atoms(f"name S6*")

N_times = u.trajectory.n_frames
N_anions = anions.n_residues
# dipole_traj = np.zeros([N_times, N_anions])
# jj=-1
# for timestep in u.trajectory:
#     jj+=1
#     for ii in range(N_anions):
#         resid = ii+1    
#         anion_res = anions.select_atoms(f"resid {resid}")
            
#         anion_center = anion_res.center_of_charge()
#         anion_dipole_vector = anion_res.dipole_vector(wrap=True)
#         anion_dipole = anion_res.dipole_moment(wrap=True)
#         dipole_traj[jj, ii] = anion_dipole
# plt.figure(1)
# for ii in range(N_anions):
#     plt.plot(dipole_traj[:,ii], label=f"resid {ii+1}")
# plt.xlabel("# frame")
# plt.ylabel("Dipole moment")

#%%
resid = 1
anion_res = anions.select_atoms(f"resid {resid}")
        
anion_center_q = anion_res.center_of_charge()
anion_center_m = anion_res.center_of_mass()
anion_dipole_vector = anion_res.dipole_vector(wrap=True, center="charge")
anion_dipole = anion_res.dipole_moment(wrap=True)

fig = plt.figure(num=resid)
ax = fig.add_subplot(projection='3d')

dipole_vector = np.zeros(3)
for atom in anion_res:
    xs, ys, zs = atom.position
    ax.scatter(xs, ys, zs, color='orange', 
               s=100*abs(atom.charge))
    dipole_vector += atom.charge*(atom.position-anion_center_q)


x, y, z = anion_center_m
ax.scatter(x,y,z,marker="x", color="k", label="center of mass")
x, y, z = anion_center_q
ax.scatter(x,y,z,marker="x", color="r", label="center of charge")
u, v, w = dipole_vector    
ax.quiver(x,y,z,u,v,w)


u2, v2, w2 = anion_dipole_vector    
ax.quiver(x,y,z,u2,v2,w2, color="red", ls='--')

ax.legend()
print(anion_dipole_vector, dipole_vector)
# %%
