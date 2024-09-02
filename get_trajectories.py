
#### TO DO: hacer script que exporte las trayectorias

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from Functions import *

path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/"

runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
MDfiles = [f"HQ.{i}" for i in range(6,11)]

dt = 0.01 # ps (tiempo entre frames) ## parametro de la simulacion: automatizar.
NPS = 2 # numero de polisulfuros

# Inicializo dataframe de trayectorias
trajs = dict.fromkeys(runs, None)
AtomTypes = [f"Li_{ii}" for ii in range(NPS*2)]+\
            [f"S6t_{ii}" for ii in range(NPS*2)]+\
            [f"S6i_{ii}" for ii in range(NPS*4)]
for key in trajs.keys():
    trajs[key] = dict.fromkeys(AtomTypes, None)

Atoms = ["Li", "S6t", "S6i"]

# loop over different runs
for idx in range(len(MDfiles)):        
    run = runs[idx]
    filename = MDfiles[idx]    
    u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.xtc")

    print("++++++++++++++++++++++++++++++++++++++++++")                             
    print(f"dataset {idx+1}/{len(runs)}")
    print(f"{path}{filename}")

    box=u.dimensions
    center = box[0:3]/2            
    # tiempo en ps
    trajectory = u.trajectory[::10]
    t = np.zeros(len(trajectory))        
    
    for Atom in Atoms:        
        group = u.select_atoms(f"name {Atom}*")
        print(f"Atoms: {Atom},   Group:  ", group)
        nAtom = -1 # indice de atomo de litio    
        for group_atom in group:
            nAtom += 1            
            atom_trajectory = np.zeros([len(trajectory), 3])
            nn = -1
            for timestep in trajectory:
                nn+=1         
                n_frame = u.trajectory.frame                    
                atom_trajectory[nn, :] = group_atom.position
            AtomType = f"{Atom}_{nAtom}"
            trajs[run][AtomType] = atom_trajectory


    # Plot trajectories:
    fig = plt.figure(idx)
    ax = fig.add_subplot(111, projection = '3d')
    for nn in range(2*NPS):
        traj = trajs[run][f"Li_{nn}"]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 
                marker = 'o', ms=1, 
                label = f'$Li_{nn}$', ls='')
    for nn in range(2*NPS):
        traj = trajs[run][f"S6t_{nn}"]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 
                marker = 'o', ms=1, ls='',
                color="goldenrod")
    for nn in range(4*NPS):
        traj = trajs[run][f"S6i_{nn}"]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], 
                marker = 'o', ms=1, ls='',
                color="gold")        

