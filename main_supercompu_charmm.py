#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  10 12024

@author: santi

read MD data and calculate ACF functions
"""


import numpy as np
from Functions import get_EFG_data, calculate_ACF, plot_ACF
import time

solvents = ["Diglyme", "ACN", "TEGDME", "DME", "DOL"] # as in folders
solvents_itp = ["DIG", "ACN", "TGD", "DME", "DIOL"] # as in itp
solvents_id = ["DIG", "ACN", "TGD", "DME", "DIOL"] # as in resname

for solvent, solvent_itp, solvent_id in zip(solvents, solvents_itp, solvents_id):
    print("="*50)
    print("="*24, solvent_id, "="*23)   
    forcefield = "charmm36.ff"
    # DME - CHARMM36 NoAnion
    path_Gromacs = fr"C:/Users/Usuario/Documents/SantiM/MDdata/mendieta/CHARMM/Li_no-anion/{solvent}_no-anion/"
    path_MDrelax = fr"C:/Users/Usuario/Documents/SantiM/MDdata/MDrelax_results/CHARMM/Li_no-anion/{solvent}_no-anion/"
    cation_itp, anion_itp, solvent_itp = [f"LIT_{solvent_itp}","none", solvent_itp] # as in .itp files
    cation, anion, solvent = ["LIT","none", solvent_id] # names
    salt = r"Li$^+$"
    Ncations = 1   # numero de Li+
    runs_inds = range(6,16) # range(6,11)
    mdp_prefix = "HQ"
    runs_prefix = "HQ"
    runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
    runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
    trajectory_format = ".xtc" # ".trr" or ".xtc"
    topology_format = ".tpr" # ".tpr" or ".gro"

    t0 = time.time()
    get_EFG_data(path_Gromacs, path_MDrelax,
                species = [cation, anion, solvent],
                species_itp = [cation_itp, anion_itp, solvent_itp],
                Ncations = Ncations,
                runs_prefix = runs_prefix,
                runs_suffix = runs_suffix,
                runs_suffix_gro = runs_suffix_gro,
                trajectory_format = trajectory_format,
                topology_format = topology_format,
                forcefield=forcefield,
                mdp_prefix=mdp_prefix
                #  rcut=0.9 # nm
                )
    print(f"EFG time: {time.time()-t0:.0f} s")


    t0 = time.time()
    calculate_ACF(path_MDrelax,
                savepath = path_MDrelax,
                species = [cation, anion, solvent],              
                Ncations = Ncations,
                runs_prefix = runs_prefix,
                runs_suffix = runs_suffix)
                #method='manual')             
    print(f"ACF time: {time.time()-t0:.0f} s")


    t0 = time.time()
    plot_ACF(path_MDrelax,
                savepath = path_MDrelax,
                species = [cation, anion, solvent],              
                salt = salt,
                Ncations = Ncations,
                runs_prefix = runs_prefix,
                runs_suffix = runs_suffix,
                max_tau = 500,
                fignum=int(np.random.random()*1e10) # to avoid superposition of graphs
                )             
    print(f"plots time: {time.time()-t0:.0f} s")
