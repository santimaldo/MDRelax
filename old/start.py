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
import pandas as pd


def get_Charges(filename):
    """    
    Parameters
    ----------
    filename : string
        The .gro gromacs autput file with x,y,z positions, and the charge in 
        thefollowing column

    Returns
    -------
    Charges : dictionary {'atom_type' : charge}        
    """    
    df =pd.read_csv(filename, delim_whitespace=True, skiprows=2, 
                    index_col=False, header=None, 
                    names=['molecule', 'atom', 'n', 'x', 'y', 'z', 'q'])
    # drop the last row (box information)
    df.drop(df.tail(1).index,inplace=True)
    
    atoms = df['atom'].unique()
    Charges = {}
    for atom in atoms:
        dftmp = df[df['atom']==atom]
        Charges[atom] = dftmp['q'].unique()[0]
    return Charges

def get_Time(filename):
    
    with open(filename, 'r') as file:
        first_line = file.readline()        
    t = first_line.split()[-3]    
    try:
        return float(t)
    except:
        msg = f"Warning: First line of '{filename}': \n"\
              f"   {first_line}"\
              f"might not in the right formaf:\n"\
              f"   system_name t=  <time> step= <Nstep>"
        print(msg)        
        return 0
#%%
# Load the GROMACS .gro file
filename = "../testsfiles/1LiCl.gro"
# filename = "../TEGDME/0fs.gro"
u = mda.Universe(filename)

#%%
# Select atoms for distance calculation (replace these selections with your specific atom names or IDs)
group_Li = u.select_atoms("name Li")
group_Cl = u.select_atoms("name Cl")

selected_atoms = u.select_atoms("all")
for atom in selected_atoms:
    atom_id = atom.id
    atom_name = atom.name
    x, y, z = atom.position
    print(f"Atom ID: {atom_id}, Atom Name: {atom_name}, Coordinates (x, y, z): {atom.position}")

# Calculate distances

# Calculate components of distance using distance_array
# x_versor, y_versor, z_versor  = np.eye(3)
# box=u.dimensions
# r_distances = mda.lib.distances.distance_array(group_Li, group_Cl,box=box)
# x_distances = mda.lib.distances.distance_array(group_Li.positions*x_versor,
#                                                 group_Cl.positions*x_versor,
#                                                 box=box)
# y_distances = mda.lib.distances.distance_array(group_Li.positions*y_versor,
#                                                 group_Cl.positions*y_versor,
#                                                 box=box)
# z_distances = mda.lib.distances.distance_array(group_Li.positions*z_versor,
#                                                 group_Cl.positions*z_versor,
#                                                 box=box)
# print("  distances:  \n", r_distances)
# print("x_distances:  \n", x_distances)
# print("y_distances:  \n", y_distances)
# print("z_distances:  \n", z_distances)


## Calculate distances

## Calculate components of distance using distance_array
x_versor, y_versor, z_versor  = np.eye(3)
box=u.dimensions
r_distances = mda.lib.distances.distance_array(group_Li, group_Cl,box=box)
r_distances_nobox = mda.lib.distances.distance_array(group_Li, group_Cl)

print(r_distances_nobox)
print(r_distances)

center = box[0:3]/2
for Li_atom in group_Li:                   
    # Put Li in the center of the universe
    Li_in_center = Li_atom.position - center
    # Redefine coordinates with respect to lithium:
    group_Cl_newpositions = group_Cl.positions - Li_in_center
    
    # apply PBC to keep the atoms within a unit-cell-length to lithiu
    group_Cl_newpositions = mda.lib.distances.apply_PBC(group_Cl_newpositions,
                                                        box,
                                                        backend='openMP')
r_distances = mda.lib.distances.distance_array(center, group_Cl_newpositions)


x_distances, y_distances, z_distances = (group_Cl_newpositions-center).T


print("  distances:  \n", r_distances)
print("x_distances:  \n", x_distances)
print("y_distances:  \n", y_distances)
print("z_distances:  \n", z_distances)



#%%
    
# x_distances = mda.lib.distances.distance_array(group_Li.positions*x_versor,
#                                                 group_Cl.positions*x_versor,
#                                                 box=box)
# y_distances = mda.lib.distances.distance_array(group_Li.positions*y_versor,
#                                                 group_Cl.positions*y_versor,
#                                                 box=box)
# z_distances = mda.lib.distances.distance_array(group_Li.positions*z_versor,
#                                                 group_Cl.positions*z_versor,
#                                                 box=box)
# print("  distances:  \n", r_distances)
# print("x_distances:  \n", x_distances)
# print("y_distances:  \n", y_distances)
# print("z_distances:  \n", z_distances)


#%%

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