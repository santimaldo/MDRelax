#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 22  2024

@author: santi

Vamos a analizar los 1H en la primera esfera de solvatacion del Li
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
#from MDAnalysis.lib.distances import distance_array
import pandas as pd
import nmrformd as nmrmd
import time


from scipy.integrate import cumulative_trapezoid


# Output:
# 'Time elapsed: X seconds'


path = "../DATA/2023-12_DME/500ps/frames_HQ_2/"
savepath = "../DATA/2023-12_DME/results/1H/"
u = mda.Universe(path+"HQ_npt-500ps_2.tpr", path+"HQ_npt-500ps_2.xtc")

#path = "../DATA/2023-12_TEGDME/500ps/frames_HQ_1/"
#u = mda.Universe(path+"HQ_npt-500ps_1.tpr", path+"HQ_npt-500ps_1.xtc")


dt = 0.01 # ps
ni = 40 # "number_i"
start_time = time.time()
end_time = time.time()

# Select H and Li atoms
H_group = u.select_atoms("name H*")
Li_group = u.select_atoms("name Li*")

## selecciono los atomos de H:
# Define a distance threshold (e.g., 5.0 Å)
distance_threshold = 3.8
# Compute the distance array
distances = mda.lib.distances.distance_array(H_group, Li_group)
# Create a mask for atoms within the distance threshold
within_threshold = distances < distance_threshold
# Determine which H atoms are within the threshold
H_group_within_threshold = H_group[np.any(within_threshold, axis=1)]
# Create a new AtomGroup called H_bond with the selected atoms
H_bond = u.select_atoms('index ' + ' '.join(map(str, H_group_within_threshold.indices)))
print("Number of selected H_bond atoms:", len(H_bond))

#%%
box = u.dimensions
center = box[0:3]/2
distances_t = []
times = []
# Iterate over each frame in the trajectory
for ts in u.trajectory:
    times.append(ts.time)    
    distances = []
    for Li_atom in Li_group:
        # Calculate distances------------------
        # Put Li in the center of the universe
        Li_in_center = Li_atom.position - center
        # Redefine coordinates with respect to lithium:
        newpositions = H_bond.positions - Li_in_center    
        # apply PBC to keep the atoms within a unit-cell-length to lithium
        H_bond_newpositions = mda.lib.distances.apply_PBC(newpositions,
                                                        box=box,
                                                        backend='openMP')
                    
        distances_to_Li = mda.lib.distances.distance_array(center, H_bond_newpositions)
        distances.append(distances_to_Li[0,:])
    distances_t.append(distances)
    
#%%
distances_t = np.array(distances_t)
# me quedo con el minimo pues es el que esta cerca del Li
distances_t = np.min(distances_t, axis=1)
times = np.array(times)


#%%
# Plot the number of H_bond atoms over time
for ii in range(len(H_bond)):
    plt.plot(times, distances_t[:,ii])
plt.xlabel('Time (ps)')
plt.ylabel('Distance (Ang)')
plt.show()
# %%
