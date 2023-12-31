#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:33:25 2023

@author: santi

First script to test the EFG summation

Arme un sistema con random walks de muchas particulas con carga -q
y un Litio "particula A" con carga +1

No hay interacciones en el movimiento, solo son random walks..
Es para ver si la autocorrelacion decae exponencialmente
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
path = "../testsfiles/randomwalk/"
filename = f"{path}randomwalk_0fs.gro"
u = mda.Universe(filename)
box=u.dimensions
center = box[0:3]/2
Charges = get_Charges(filename)




# times = np.arange(11)*10
num_steps = 1000
dt = 0.01  # ps
times = np.arange(num_steps)*dt*1000 # fs
t = np.zeros(times.size)

EFG = []
Li_positions = []
for ii in range(times.size):        
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
    # Load the GROMACS .gro file
    filename = f"{path}randomwalk_{times[ii]:.0f}fs.gro"
    t[ii] = get_Time(filename)
    print(f"       time = {t[ii]/1000} ps\n\n")
    print(filename)
    u = mda.Universe(filename)
    box=u.dimensions
    center = box[0:3]/2
    Charges = get_Charges(filename)

    group_Li = u.select_atoms("name A")    
    Li_pos_t = []    
    EFG_t = []
    nLi = -1 # indice de atomo de litio    
    for Li_atom in group_Li:
        nLi += 1
        Li_pos_t.append(Li_atom.position)
        EFG_t_nLi = 0
        for AtomType in Charges['AtomType']:     
            for AtomType in Charges['AtomType']:        
                if 'A' in AtomType.lower():
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
            EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                    y_distances, z_distances)
            if np.isnan(EFG_t_AtomType).any():
                print(f"    NAN!!!!! {filename}")
                STOP
            # EFG_t = [EFG_t[kk]+EFG_t_AtomType[kk] for kk in range(6)]
            EFG_t_nLi += EFG_t_AtomType
        EFG_t.append(EFG_t_nLi)
    
    Li_positions.append(Li_pos_t)    
    EFG.append(EFG_t) # cada elemento de la lista es un tiempo
    
    
# cada columna de EFG corresponde al litio de group_Li (EN ESE ORDEN)    
EFG = np.array(EFG)
    
#%% Calculo el profucto de EFG a tiempo t y a tiempo 0
t = t - t[0]

ACF = np.zeros([t.size, group_Li.n_atoms])
for ii in range(group_Li.n_atoms):
    efg_nLi = EFG[:,ii,:,:]    
    ACF[:,ii] = np.sum(efg_nLi*efg_nLi[0,:,:], axis=(1,2))
    
    plt.figure(0)
    plt.plot(t, ACF[:,ii]/ACF[0,ii],'o-', label = rf'$Li_{ii+1}$')
plt.plot(t, np.mean(ACF, axis=1)/np.mean(ACF, axis=1)[0],'o-', label = r'mean')
plt.xlabel("Time [ps]", fontdict={'fontsize':16})
plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})
plt.legend()


plt.show()

#---------------
#%% Calculo el profucto de EFG a tiempo t y a tiempo 0,
### esta vez variando cual es el tiempo 0 (promedio en ensamble)


efg = EFG
tmax = t[-1]*1000 # paso a fs

acf = np.zeros([t.size, group_Li.n_atoms])
for ii in range(t.size):    
    tau = ii*10
    jj, t0, acf_ii = 0, 0, 0
    while t0+tau<=tmax:
        print(f"tau = {tau} fs, t0 = {t0} ps, ---------{jj}")                
        acf_ii += np.sum(efg[ii,:,:,:]*efg[jj,:,:,:], axis=(1,2))
        jj+=1
        t0 = jj*10
    print(f"el promedio es dividir por {jj}")
    acf[ii,:] = acf_ii/jj

#%%
plt.figure(44848)
for jj in range(group_Li.n_atoms):        
    # plt.plot(t, ACF[:,jj]/ACF[0,jj], 'o--', 
             # label=f'Li {jj+1}, Sin promediar')   
    plt.plot(t, acf[:,jj], 'o-', 
             label=rf'$Li_{jj+1}$')
        
plt.plot(t, np.mean(acf, axis=1),'o-', 
         label = r'mean')
plt.xlabel("Time [ps]", fontdict={'fontsize':16})

plt.ylabel(r"$\langle\ EFG(t)\cdot EFG(0)\ \rangle_t$",
           fontdict={'fontsize':16})
plt.gca().axhline(0, color='k', ls='--')
# plt.gca().axhline(1, color='k', ls='--')
plt.tight_layout()
plt.legend(loc='center right')
plt.savefig(f"{path}ACF.png")
plt.show()

#%%
plt.figure(44849)
plt.plot(t, np.mean(acf, axis=1)/np.mean(acf, axis=1)[0],'o-', color='green', 
         label = r'Promedio entre atomos de Li y en $<>_t$')
plt.xlabel("Time [ps]", fontdict={'fontsize':16})
plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})
plt.gca().axhline(0, color='k', ls='--')
plt.gca().axhline(1, color='k', ls='--')
plt.tight_layout()
plt.legend()
plt.savefig(f"{path}ACFnorm.png")
plt.show()
