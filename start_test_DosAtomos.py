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
    
    


def calculate_EFG(q, r, x, y, z):
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

    Vxx = np.sum(Vxx)
    Vxy = np.sum(Vxy)
    Vxz = np.sum(Vxz)
    Vyy = np.sum(Vyy)
    Vyz = np.sum(Vyz)
    Vzz = np.sum(Vzz)
    
    EFG = np.array([[Vxx, Vxy, Vxz],
                    [Vxy, Vyy, Vyz],
                    [Vxz, Vyz, Vzz]]) 

    return EFG
#%%
# Primero leo el tiempo 0 para establecer algunos valores generales del universo

ACFs = []
for nn in range(4):
    
    path = f"../DosAtomos/caso_{nn}/"
    
    filename = f"{path}LiCl_0fs.gro"
    u = mda.Universe(filename)
    box=u.dimensions
    center = box[0:3]/2
    Charges = get_Charges(filename)
    
    
    
    
    times = np.arange(11)*10
    t = np.zeros(times.size)
    r = np.zeros_like(t)
    
    EFG = []
    Li_positions = []
    Cl_positions = []
    for ii in range(times.size):        
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
        # Load the GROMACS .gro file
        filename = f"{path}LiCl_{times[ii]}fs.gro"
        t[ii] = get_Time(filename)
        print(f"       time = {t[ii]} ps\n\n")
        print(filename)
        u = mda.Universe(filename)
        box=u.dimensions
        center = box[0:3]/2
        Charges = get_Charges(filename)
    
        group_Li = u.select_atoms("name Li*")    
        Li_pos_t = []
        Cl_pos_t = []
        EFG_t = []
        nLi = -1 # indice de atomo de litio    
        for Li_atom in group_Li:
            nLi += 1
            Li_pos_t.append(Li_atom.position)
            EFG_t_nLi = 0
            for AtomType in Charges['AtomType']:        
                if 'li' in AtomType.lower():
                    continue # NO CALCULO ENTRE LITIOS
                
                
                q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                
                group = u.select_atoms(f"name {AtomType}")
                if 'cl' in AtomType.lower() and nLi==0:                
                    Cl_pos_t.append(group.positions)
                    
                if ii==0:
                    # distancia inicial entre atomos:
                    r0 = mda.lib.distances.calc_bonds(Li_atom.position, 
                                                      group.positions)[0]
                    # tener en cuenta que esto asume que solo hay un atomo de Li
                    # y uno de Cl
                
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
                
                r[ii] = r_distances # solo valido para DOS ATOMOS!!!
        
                x_distances, y_distances, z_distances = (group_newpositions-center).T
        
        
        
                # Calculate EFG--------------------------------------------------------
                EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                        y_distances, z_distances)
                      
                # EFG_t = [EFG_t[kk]+EFG_t_AtomType[kk] for kk in range(6)]
                EFG_t_nLi += EFG_t_AtomType
            EFG_t.append(EFG_t_nLi)
        
        Li_positions.append(Li_pos_t)
        Cl_positions.append(Cl_pos_t)    
        EFG.append(EFG_t) # cada elemento de la lista es un tiempo
        
        
    # cada columna de EFG corresponde al litio de group_Li (EN ESE ORDEN)    
    EFG = np.array(EFG)
        
    #%% Calculo el profucto de EFG a tiempo t y a tiempo 0
    t = t - t[0]
    
    ACF = np.zeros([t.size, group_Li.n_atoms])
    for ii in range(group_Li.n_atoms):
        efg_nLi = EFG[:,ii,:,:]    
        ACF[:,ii] = np.sum(efg_nLi*efg_nLi[0,:,:], axis=(1,2))
    ACFs.append(ACF)
    
        # plt.figure(0)
        # plt.plot(t, ACF[:,ii]/ACF[0,ii],'o-', label = rf'$Li_{ii+1}$')
    # plt.plot(t, np.mean(ACF, axis=1)/np.mean(ACF, axis=1)[0],'o-', label = r'mean')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    # plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})    
    # plt.legend()        
    
    

#---------------
ACFs = np.array(ACFs)
#%%

# Conozco la solucion exacta:
r0ex = 1
rex = 1 + 0.1*t*1000
acf_exacta = 6/(r0ex**3*rex**3)

labels = ['y', 'z', '-x', 'x+y+z']
ACF = np.zeros_like(t)
for jj in range(4):
    ACF[:] = ACFs[jj,:,:].T
    plt.figure(44848)
    plt.semilogy(t, ACF, 'o--', 
              label=fr'Caso {jj}, velocidad en {labels[jj]}')

    plt.figure(44849)
    plt.plot(t, (ACF.T-acf_exacta)/acf_exacta*100,
                 'o--', label=f'Li {jj+1}, Caso {jj}')
    



plt.figure(44848)    
plt.semilogy(t, acf_exacta,'kx--', label = r'Exacta')
plt.xlabel('Tiempo [ps]')
plt.ylabel('EFG(t)*EFG(0)')
plt.legend()
plt.figure(44849)
plt.title('Error porcentual')
plt.ylabel('(Calculada-exacta)/exacta*100 [%]')
plt.xlabel('Tiempo [ps]')
plt.legend()
    
