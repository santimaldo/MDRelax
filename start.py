#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:33:25 2023

@author: santi

First script to test the EFG summation
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda


# Load the GROMACS .gro file
gro_file = "../2LiCl.gro"
u = mda.Universe(gro_file)

# Select atoms for distance calculation (replace these selections with your specific atom names or IDs)
group_Li = u.select_atoms("name Li")
group_Cl = u.select_atoms("name Cl")

selected_atoms = u.select_atoms("all")
for atom in selected_atoms:
    atom_id = atom.id
    atom_name = atom.name
    x, y, z = atom.position
    print(f"Atom ID: {atom_id}, Atom Name: {atom_name}, Coordinates (x, y, z): {atom.position}")

# Calculate distances between specified atoms
r_distances = mda.lib.distances.distance_array(group_Li, group_Cl)
distances_ang = mda.lib.distances.calc_angles(group_Li.positions, group_Cl.positions)

# Calculate distances between specified atoms
r_distances = mda.lib.distances.distance_array(group_Li, group_Cl)


for Li_atom in group_Li:
distances = np.abs(group_Li.positions - group_Cl.positions)
x_distances = distances[:, 0]
y_distances = distances[:, 1]
z_distances = distances[:, 2]


r = np.array([3, 4, 12])



np.array([r,r,r])
dist = np.linalg.norm(r)


dist3inv = 1/(r_distances*r_distances*r_distances)
dist5inv = 1/((r_distances*r_distances*r_distances*r_distances*r_distances))

q = -1

Vxx = q *(3 * x_distances * x_distances * dist5inv - dist3inv)
Vxy = q *(3 * x_distances * y_distances * dist5inv )
Vxz = q *(3 * x_distances * z_distances * dist5inv )
Vyy = q *(3 * y_distances * y_distances * dist5inv - dist3inv)
Vyz = q *(3 * y_distances * z_distances * dist5inv )
Vzz = q *(3 * z_distances * z_distances * dist5inv - dist3inv)


Vsum = 2*(Vxy+Vxz+Vyz) + (Vxx+Vyy+Vzz)

# print(Vxx*1e4, Vxy*1e4, Vxz*1e4, Vyy*1e4, Vyz*1e4, Vzz*1e4)
print(Vsum)