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



def get_EFG_data(path_Gromacs, path_MDrelax,
                 species = ["cation", "anion", "solvent"],
                 species_itp = ["Li","none", "DME_7CB8A2"],
                 Ncations = 1,
                 runs_prefix = "HQ",
                 runs_suffix = None,
                 runs_suffix_gro = None,
                 trajectory_format = ".trr",
                 topology_format = ".gro"):
    """
    Function for reading GROMACS data and calculating the EFG
    at the cations' positions from different EFG-sources (other charges)
    """
    # Unpacking variables:
    cation, anion, solvent = species # names
    cation_itp, anion_itp, solvent_itp = species_itp # as in .itp files
        
    runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]    
    runs_gro = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix_gro]    
    #---------------------------------------------------
    # get_dt() takes the .mdp file
    dt = get_dt(f"{path_Gromacs}{runs_prefix}.mdp")
    Charges = get_Charges([cation_itp, anion_itp, solvent_itp], path_Gromacs)

    # loop over different runs
    for idx in range(len(runs)):    
        run = runs[idx]
        run_gro = runs_gro[idx]
        u = mda.Universe(f"{path_Gromacs}{run_gro}{topology_format}",
                         f"{path_Gromacs}{run_gro}{trajectory_format}")    
        print(u)                         
        box=u.dimensions
        center = box[0:3]/2
                
        # tiempo en ps    
        t = np.arange(len(u.trajectory))*dt
        # Arreglo EFG:
        # EFG.shape --> (NtimeSteps, Ncations, 3, 3)        
        EFG_anion = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        EFG_cation = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        EFG_solvent = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        t_ind = -1
        for timestep in u.trajectory:
            t_ind+=1         
            n_frame = u.trajectory.frame
            if t_ind%100==0:                
                print("++++++++++++++++++++++++++++++++++++++++++")
                print(f"dataset {idx+1}/{len(runs)}, frame={n_frame}, time = {t[t_ind]:.2f} ps\n\n")                

            cations_group = u.select_atoms(f"name {cation}*")
            
            cation_index = -1 # indice de atomo de litio    
            for cation_in_group in cations_group:
                cation_index += 1            
                # aqui guarfo el EFG sobre el n-esimo Li al tiempo t:
                EFG_t_nthcation_anion = np.zeros([3,3])
                EFG_t_nthcation_cation = np.zeros([3,3])
                EFG_t_nthcation_solvent = np.zeros([3,3])
                for residue, AtomType in zip(Charges['residue'],
                                            Charges['AtomType']):
                    q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                    
                    group = u.select_atoms(f"name {AtomType}")                               
                    if group.n_atoms==0: continue
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
                        
                    if anion in residue:
                        EFG_t_nthcation_anion += EFG_t_AtomType                    
                    elif cation in residue:
                        EFG_t_nthcation_cation += EFG_t_AtomType
                    else:
                        EFG_t_nthcation_solvent += EFG_t_AtomType                    
                EFG_anion[t_ind, cation_index, :, :] = EFG_t_nthcation_anion
                EFG_cation[t_ind, cation_index, :, :] = EFG_t_nthcation_cation
                EFG_solvent[t_ind, cation_index, :, :] = EFG_t_nthcation_solvent
        #---------------------------------------------------------
        ### EXPORT DATA        
        EFG_total = EFG_cation + EFG_anion + EFG_solvent
        EFGs = [EFG_cation, EFG_anion, EFG_solvent, EFG_total]
        efg_sources = [cation, anion, solvent, "total"]
        for EFG, efg_source in zip(EFGs, efg_sources):
            Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
            Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
            
            data = [t]
            for nn in range(cations_group.n_atoms):    
                data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
            
            data = np.array(data).T    
            header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
                    r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
            filename = f"{path_MDrelax}/EFG_{efg_source}_{run}.dat"
            np.savetxt(filename, data, header=header)
        del EFG_anion, EFG_solvent
        #----------------------------------------------------------    
    return 0