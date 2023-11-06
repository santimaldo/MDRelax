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
    Charges : dataFrame ['AtomType',  'Charge']
    """    
    df =pd.read_csv(filename, delim_whitespace=True, skiprows=2, 
                    index_col=False, header=None, 
                    names=['molecule', 'atom', 'n', 'x', 'y', 'z', 'q'])
    # drop the last row (box information)
    df.drop(df.tail(1).index,inplace=True)
    
    atoms = df['atom'].unique()
    Charges = []
    for atom in atoms:
        df_tmp = df[df['atom']==atom]
        q = df_tmp['q'].unique()[0]
        Charges.append([atom, q])
    Charges = pd.DataFrame(Charges, columns=['AtomType','Charge'])
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
    
    


def calculate_EFG(q, r, x, y, z, return_array=False):
    """
    Calculate de EFG based on charge points.
    r is the distance between observation point and EFG source (q).
    x, y, z are the components of vec(r).
    """
    dist3inv = 1/(r*r*r)
    dist5inv = 1/((r*r*r*r*r))
                    
    Vxx = q *(3 * x * x * dist5inv - dist3inv)
    Vxy = q *(3 * x * y * dist5inv )
    Vxz = q *(3 * x * z * dist5inv )
    Vyy = q *(3 * y * y * dist5inv - dist3inv)
    Vyz = q *(3 * y * z * dist5inv )
    Vzz = q *(3 * z * z * dist5inv - dist3inv)

    Vsum = 2*(Vxy+Vxz+Vyz) + (Vxx+Vyy+Vzz)    
    EFG = np.zeros(3)
    if return_array:
        EFG = np.array([[Vxx, Vxy, Vxz],
                        [Vxy, Vyy, Vyz],
                        [Vxz, Vyz, Vzz]])                
    return EFG, Vsum
#%%

times = np.arange(11)*10
t = np.zeros(times.size)
EFG =np.zeros([times.size, 2])
#%%
for ii in range(times.size):
    # Load the GROMACS .gro file
    filename = f"../TEGDME/{times[ii]}fs.gro"
    t[ii] = get_Time(filename)
    print(filename)
    u = mda.Universe(filename)
    box=u.dimensions
    center = box[0:3]/2
    Charges = get_Charges(filename)

    group_Li = u.select_atoms("name Li*")

    Vsum_Li = []
    for Li_atom in group_Li:
        Vsum = 0                       
        for AtomType in Charges['AtomType']:        
            if 'li' in AtomType.lower():
                continue # NO CALCULO ENTRE LITIOS
            q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
            
            group = u.select_atoms(f"name {AtomType}")
            
            # Calculate distances------------------
            # Put Li in the center of the universe
            Li_in_center = Li_atom.position - center
            # Redefine coordinates with respect to lithium:
            group_newpositions = group.positions - Li_in_center
        
            # apply PBC to keep the atoms within a unit-cell-length to lithiu
            group_Cl_newpositions = mda.lib.distances.apply_PBC(group_newpositions,
                                                                box=box,
                                                                backend='openMP')
            
            r_distances = mda.lib.distances.distance_array(center, 
                                                           group_newpositions)
    
            x_distances, y_distances, z_distances = (group_newpositions-center).T
    
    
    
            # Calculate EFG--------------------------------------------------------
            _, Vsum_atoms = calculate_EFG(q, r_distances, x_distances,
                                    y_distances, z_distances)
                  
            Vsum += np.sum(Vsum_atoms)                            
        Vsum_Li.append(Vsum)
    
        
    EFG[ii] = Vsum_Li
    
#%%

t = t - t[0]
ACF = EFG*EFG[0,:]


plt.plot(t, ACF[:,0], 'o-')        
plt.plot(t, ACF[:,1], 'o-')        
plt.show()

#---------------
#%%

efg = np.sum(EFG, axis=1)*np.sum(EFG[0,:])

acf = np.zeros_like(efg)
for ii in range(efg.size):    
    tau = ii*10
    jj, t0, acf = 0
    while t0+tau<=100:
        print(f"tau = {tau} fs, t0 = {t0} ps")
        acf += efg[ii]*efg[jj]
        jj+=1
        t0 = jj*10
    print(f"el promedio es dividir por j")
    acf[ii] = 
    
    
