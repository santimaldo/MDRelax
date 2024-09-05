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
from Functions import *

path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/"
savepath = path+"MDRelax/"
cation, anion, solvent = ["Li","S6", "DME_7CB8A2"] # as in .itp files
solvent_hr = "DME"

runs_inds = range(6,11)
mdp_file = "HQ"
MDfiles = [f"HQ.{i}" for i in runs_inds]
runs = [f"{t*1000:.0f}_ps" for t in runs_inds]

trajectory_format = ".trr" # ".trr" or ".xtc"
topology_format = ".gro" # ".tpr" or ".gro"


# path = "/home/santi/mendieta/TEGDME/"
# savepath = "/home/santi/MD/MDRelax_results/TEGDME/"
# cation, anion, solvent = ["Li","S6", "tegdme"] # as in .itp files
# solvent_hr = "TEGDME"
# runs = [f"{t:.0f}_ps" for t in [6000,7000,8000,9000,10000]]
# MDfiles = [f"HQ.{i}" for i in range(6,11)]


Ncations = 4 # numero de Li+
dt = get_dt(mdp_file=f"{path}{mdp_file}.mdp")
Charges = get_Charges([cation, anion, solvent], path)

# loop over different runs
for idx in range(len(MDfiles)):    
    run = runs[idx]
    filename = MDfiles[idx]    
    u = mda.Universe(f"{path}{filename}{topology_format}", f"{path}{filename}{trajectory_format}")    
    box=u.dimensions
    center = box[0:3]/2
            
    # tiempo en ps    
    t = np.arange(len(u.trajectory))*dt
    # Arreglo EFG:
    # EFG.shape --> (NtimeSteps, Ncations, 3, 3)        
    EFG_anion = np.zeros([len(u.trajectory), Ncations, 3, 3])    
    EFG_cation = np.zeros([len(u.trajectory), Ncations, 3, 3])    
    EFG_solvent = np.zeros([len(u.trajectory), Ncations, 3, 3])    
    nn = -1
    for timestep in u.trajectory:
        nn+=1         
        n_frame = u.trajectory.frame
        if nn%100==0:                
            print("++++++++++++++++++++++++++++++++++++++++++")
            print(f"dataset {idx+1}/{len(MDfiles)}, frame={n_frame}, time = {t[nn]:.2f} ps\n\n")                

        cations_group = u.select_atoms(f"name {cation}*")
        
        cation_index = -1 # indice de atomo de litio    
        for cation_in_group in cations_group:
            cation_index += 1            
            # aqui guarfo el EFG sobre el n-esimo Li al tiempo t:
            EFG_t_nthcation_anion = np.zeros([3,3])
            EFG_t_nthcation_cation = np.zeros([3,3])
            EFG_t_nthcation_solvent = np.zeros([3,3])
            for AtomType in Charges['AtomType']:
                q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                
                group = u.select_atoms(f"name {AtomType}")                                
                # Calculate distances------------------
                # Put Li in the center of the universe
                cation_in_center = cation_in_group.position - center
                # Redefine coordinates with respect to lithium:
                group_newpositions = group.positions - cation_in_center
                # apply PBC to keep the atoms within a unit-cell-length to lithiu
                group_newpositions_pbc = mda.lib.distances.apply_PBC(group_newpositions,
                                                                    box=box,
                                                                    backend='openMP')
                # The distances to the nth-cation is the distance from the center:
                r_distances = mda.lib.distances.distance_array(center,
                                                               group_newpositions_pbc,
                                                               backend='openMP')[0,:]
                x_distances, y_distances, z_distances = (group_newpositions_pbc-center).T
                
                if cation in AtomType:    
                    # Quito la distancia cero, i.e, entre la "autodistancia"
                    # correccion:
                    #    Cambio:
                    #       x_distances = x_distances[r_distances!=0]
                    #    por:
                    #       x_distances = x_distances[r_distance > 1e-5]
                    # De esta manera, me quito de encima numeros que son
                    # distintos de 0 por error numerico
                    x_distances = x_distances[r_distances>1e-5]
                    y_distances = y_distances[r_distances>1e-5]
                    z_distances = z_distances[r_distances>1e-5]
                    r_distances = r_distances[r_distances>1e-5]
            
                if (r_distances<1).any():
                    msg = f"hay un {AtomType} a menos de 1 A del {cation}_{cation_index}"
                    raise Warning(msg)
                # Calculate EFG
                EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                        y_distances, z_distances)
                      
                if anion in AtomType:
                    EFG_t_nthcation_anion += EFG_t_AtomType
                if cation in AtomType:
                    EFG_t_nthcation_cation += EFG_t_AtomType
                else:
                    EFG_t_nthcation_solvent += EFG_t_AtomType                    
            EFG_anion[nn, cation_index, :, :] = EFG_t_nthcation_anion
            EFG_cation[nn, cation_index, :, :] = EFG_t_nthcation_cation
            EFG_solvent[nn, cation_index, :, :] = EFG_t_nthcation_solvent
    #---------------------------------------------------------
    ### EXPORT DATA        
    for EFG, efg_source in zip([EFG_cation, EFG_anion, EFG_solvent], [cation, anion, solvent_hr]):
        Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
        Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
        
        data = [t]
        for nn in range(cations_group.n_atoms):    
            data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
        
        data = np.array(data).T    
        header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
                r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
        filename = f"{savepath}/EFG_{efg_source}_{run}.dat"
        np.savetxt(filename, data, header=header)
    del EFG_anion, EFG_solvent
    #----------------------------------------------------------    
    
    