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

#%%
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


frame_times = ["500ps", "1ns"]
runs = ["frames_HQ_1", "frames_HQ_2"]
# path = "../DATA/2023-12_TEGDME/500ps/frames_HQ_1/"
path = f"../GromacsFiles/2024-08-15_DME_2nd-test/"
filename_format = ".fs.gro" # los archivos se llaman: <time>{filename_format}


for frame_time in frame_times:
  for run in runs:

    filename = f"{path}{frame_time}/{run}/0{filename_format}"    
    u = mda.Universe(filename)
    box=u.dimensions
    center = box[0:3]/2
    Charges = get_Charges(filename)
    
    
    # times = np.arange(11)*10
    times = np.arange(3001)*10
    t = np.zeros(times.size)
    
    EFG = []
    Li_positions = []
    St_positions = []
    Si_positions = []
    for ii in range(times.size):        
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
        # Load the GROMACS .gro file    
        filename = f"{path}{frame_time}/{run}/{times[ii]}{filename_format}"
        t[ii] = get_Time(filename)
        print(f"       time = {t[ii]} ps\n\n")
        print(filename)
        u = mda.Universe(filename)
        box=u.dimensions
        center = box[0:3]/2
        Charges = get_Charges(filename)
    
        group_Li = u.select_atoms("name Li*")    
        Li_pos_t = []
        St_pos_t = []
        Si_pos_t = []
        EFG_t = []
        nLi = -1 # indice de atomo de litio    
        for Li_atom in group_Li:
            nLi += 1
            Li_pos_t.append(Li_atom.position)
            EFG_t_nLi = 0
            for AtomType in Charges['AtomType']:        
                
                    
                q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                
                group = u.select_atoms(f"name {AtomType}")
                if 's6t' in AtomType.lower() and nLi==0:                
                    St_pos_t.append(group.positions)
                elif 's6i' in AtomType.lower() and nLi==0:                
                    Si_pos_t.append(group.positions)
                    
                
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
        
                if 'li' in AtomType.lower():
                     # Quito la distancia cero, i.e, entre la "autodistancia"
                     x_distances = x_distances[x_distances!=0]
                     y_distances = y_distances[y_distances!=0]
                     z_distances = z_distances[z_distances!=0]
                     r_distances = r_distances[r_distances!=0]
        
                # Calculate EFG--------------------------------------------------------
                EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                        y_distances, z_distances)
                      
                # EFG_t = [EFG_t[kk]+EFG_t_AtomType[kk] for kk in range(6)]
                EFG_t_nLi += EFG_t_AtomType
            EFG_t.append(EFG_t_nLi)
        
        Li_positions.append(Li_pos_t)
        St_positions.append(St_pos_t)
        Si_positions.append(Si_pos_t)
        EFG.append(EFG_t) # cada elemento de la lista es un tiempo
        
        
    # cada columna de EFG corresponde al litio de group_Li (EN ESE ORDEN)    
    EFG = np.array(EFG)
    # EFG.shape --> (NtimeSteps, NLiAtoms, 3, 3)
    #%%
    ### EXPORT DATA    
    Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
    Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
    
    data = [t]
    for nn in range(group_Li.n_atoms):    
        data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
    
    data = np.array(data).T
    #%%
    header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
              r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
    filename = f"{path}EFG_{frame_time}_{run}.dat"
    np.savetxt(filename, data, header=header)
        
    # #%% Calculo el profucto de EFG a tiempo t y a tiempo 0
    # t = t - t[0]
    
    # ACF = np.zeros([t.size, group_Li.n_atoms])
    # for ii in range(group_Li.n_atoms):
    #     efg_nLi = EFG[:,ii,:,:]    
    #     ACF[:,ii] = np.sum(efg_nLi*efg_nLi[0,:,:], axis=(1,2))
        
    #     plt.figure(0)
    #     plt.plot(t, ACF[:,ii]/ACF[0,ii],'o-', label = rf'$Li_{ii+1}$')
    # plt.plot(t, np.mean(ACF, axis=1)/np.mean(ACF, axis=1)[0],'o-', label = r'mean')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    # plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})
    # plt.gca().axhline(0, color='k', ls='--')
    # plt.legend()
    
    
    # plt.show()
    
    # #---------------
    # #%% Calculo el profucto de EFG a tiempo t y a tiempo 0,
    # ### esta vez variando cual es el tiempo 0 (promedio en ensamble)
    
    # efg = EFG
    # acf = np.zeros([t.size, group_Li.n_atoms])
    # Num_promedios = np.zeros(t.size)
    # for ii in range(t.size):    
    #     tau = ii*10
    #     jj, t0, acf_ii = 0, 0, 0
    #     while t0+tau<=times[-1]:
    #         print(f"tau = {tau} fs, t0 = {t0} ps, ---------{jj}")                
    #         acf_ii += np.sum(efg[ii,:,:,:]*efg[ii+jj,:,:,:], axis=(1,2))
    #         jj+=1
    #         t0 = jj*10
    #     print(f"el promedio es dividir por {jj}")
    #     acf[ii,:] = acf_ii/jj
    #     Num_promedios[ii] = jj
    # #%%
    # plt.figure(44847)
    # plt.plot(t, Num_promedios,'k--')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    
    # plt.ylabel(r"Numero de promedios",
    #            fontdict={'fontsize':16})
    
    # plt.figure(44848)
    # for jj in range(group_Li.n_atoms):        
    #     # plt.plot(t, ACF[:,jj]/ACF[0,jj], 'o--', 
    #              # label=f'Li {jj+1}, Sin promediar')   
    #     plt.plot(t, acf[:,jj], 'o-', 
    #              label=rf'$Li_{jj+1}$')
            
    # plt.plot(t, np.mean(acf, axis=1),'o-', 
    #          label = r'mean')
    
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    
    # plt.ylabel(r"$\langle\ EFG(t)\cdot EFG(0)\ \rangle_t$",
    #            fontdict={'fontsize':16})
    
    # # plt.xlim([0,3])
    # plt.gca().axhline(0, color='k', ls='--')
    # # plt.gca().axhline(1, color='k', ls='--')
    # plt.tight_layout()
    # plt.legend(loc='center right')
    # plt.savefig(f"{path}ACF.png")
    # plt.show()
    
    # #%%
    # plt.figure(44849)
    # plt.plot(t, np.mean(acf, axis=1)/np.mean(acf, axis=1)[0],'o-', color='green', 
    #          label = r'Promedio entre atomos de Li y en $<>_t$')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    # plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})
    # plt.gca().axhline(0, color='k', ls='--')
    # plt.gca().axhline(1, color='k', ls='--')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f"{path}ACFnorm.png")
    # plt.show()
    
    