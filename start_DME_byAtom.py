#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:33:25 2023

@author: santi

First script to test the EFG summation


TO DO LIST:
hacer que obtenga el tiempo de la simulacion
"""
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from Functions import *

# path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/"
# savepath = path
# species_list = ["Li", "S6", "DME_7CB8A2"]

path = "/home/santi/mendieta/TEGDME/"
savepath = "/home/santi/MD/MDRelax_results/TEGDME/"
species_list = ["Li", "S6", "tegdme"]

runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
MDfiles = [f"HQ.{i}" for i in range(6,11)]

NPS = 2 # moleculas de polisulfuro
dt = 0.01 # ps (tiempo entre frames) ## parametro de la simulacion: automatizar.

Charges = get_Charges(species_list, path)

# loop over different runs
for idx in range(len(MDfiles)):    
    run = runs[idx]
    filename = MDfiles[idx]    
    u = mda.Universe(f"{path}{filename}.tpr", f"{path}{filename}.trr")
    #u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.trr")
    # u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.xtc")
    box=u.dimensions
    center = box[0:3]/2
            
    # tiempo en ps
    trajectory = u.trajectory        
    t = np.arange(len(trajectory))*dt
    # Arreglo EFG:
    # EFG.shape --> (NtimeSteps, NLiAtoms, 3, 3)        
    EFG_sulfur = np.zeros([len(trajectory), NPS*2, 3, 3])    
    EFG_solvent = np.zeros([len(trajectory), NPS*2, 3, 3])    
    nn = -1
    for timestep in trajectory:
        nn+=1 
        print("++++++++++++++++++++++++++++++++++++++++++")                    
        n_frame = u.trajectory.frame                
        print(f"dataset {idx+1}/{len(MDfiles)}, frame={n_frame}, time = {t[nn]:.2f} ps\n\n")                

        group_Li = u.select_atoms("name Li*")                    
        #EFG_t = pd.DataFrame[columns=Charges['AtomType']] ############### CONTINUAR ESTA IDEA
        nLi = -1 # indice de atomo de litio    
        for Li_atom in group_Li:
            nLi += 1            
            # aqui guarfo el EFG sobre el n-esimo Li al tiempo t:
            EFG_t_nLi_sulfur = np.zeros([3,3])
            EFG_t_nLi_solvent = np.zeros([3,3])
            for AtomType in Charges['AtomType']:                                           
                q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                
                group = u.select_atoms(f"name {AtomType}")                                
                # Calculate distances------------------
                # Put Li in the center of the universe
                Li_in_center = Li_atom.position - center
                # Redefine coordinates with respect to lithium:
                group_newpositions = group.positions - Li_in_center            
                # apply PBC to keep the atoms within a unit-cell-length to lithiu
                group_newpositions = mda.lib.distances.apply_PBC(group_newpositions,
                                                                    box=box,
                                                                    backend='openMP')                                                            
                if 'li' not in AtomType.lower():
                    r_distances = mda.lib.distances.distance_array(center, 
                                                               group_newpositions)        
                    x_distances, y_distances, z_distances = (group_newpositions-center).T                      
                else:
                    # Defino la distancia de otra forma, para poder quitar
                    # la "autodistancia" del litio observado                    
                    x_distances, y_distances, z_distances = (group_newpositions-center).T                     
                    r_distances = np.sqrt(x_distances*x_distances+
                                          y_distances*y_distances+
                                          z_distances*z_distances)
                    # Quito la distancia cero, i.e, entre la "autodistancia"
                    x_distances = x_distances[r_distances!=0]
                    y_distances = y_distances[r_distances!=0]
                    z_distances = z_distances[r_distances!=0]
                    r_distances = r_distances[r_distances!=0]
        
                # Calculate EFG--------------------------------------------------------
                EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                        y_distances, z_distances)
                      
                if 's6' in AtomType.lower():
                    EFG_t_nLi_sulfur += EFG_t_AtomType
                else:
                    EFG_t_nLi_solvent += EFG_t_AtomType                    
            EFG_sulfur[nn, nLi, :, :] = EFG_t_nLi_sulfur
            EFG_solvent[nn, nLi, :, :] = EFG_t_nLi_solvent
    #---------------------------------------------------------
    ### EXPORT DATA        
    for EFG, efg_source in zip([EFG_sulfur, EFG_solvent], ["sulfur", "solvent"]):
        Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
        Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
        
        data = [t]
        for nn in range(group_Li.n_atoms):    
            data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
        
        data = np.array(data).T    
        header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
                r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
        filename = f"{savepath}/EFG_{efg_source}_{run}.dat"
        np.savetxt(filename, data, header=header)
    del EFG_sulfur, EFG_solvent
    #----------------------------------------------------------    
    
    