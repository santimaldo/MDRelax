
#### TO DO: hacer script que exporte las trayectorias

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from Functions import *

path = "/home/santi/MD/GromacsFiles/2024-08-15_DME_2nd-test/"

runs = [f"{t:.1f}_ps" for t in [0.5,1,1.5,2]]
MDfiles = [f"HQ.{i}" for i in range(6,10)]


dt = 0.01 # ps (tiempo entre frames) ## parametro de la simulacion: automatizar.

# loop over different runs
for idx in range(len(MDfiles)):    
    run = runs[idx]
    filename = MDfiles[idx]    
    # u = mda.Universe(f"{path}{filename}.tpr", f"{path}{filename}.trr")
    u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.trr")
    box=u.dimensions
    center = box[0:3]/2
            
    # tiempo en ps
    trajectory = u.trajectory[:10000:5]
    t = np.zeros(len(trajectory))

    if idx==0:
         # aca va a estar impuesto de antemano el orden de las columnas:
         # Li Li S6t S6t S6i S6i S6i S6i
         PS_trajectory = np.zeros([len(trajectory), 8])
         
    nn = -1
    for timestep in trajectory:
        nn+=1 
        print("++++++++++++++++++++++++++++++++++++++++++")                    
        n_frame = u.trajectory.frame        
        t[nn] = n_frame * dt
        print(f"dataset {idx}, frame={n_frame}, time = {t[nn]:.2f} ps\n\n")                

        group_Li = u.select_atoms("name Li*")                   
        nLi = -1 # indice de atomo de litio    
        for Li_atom in group_Li:
            nLi += 1            
            PS_trajectory[nn,nLi] = Li_atom.position
            