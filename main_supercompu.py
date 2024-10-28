#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  10 12024

@author: santi

read MD data and calculate ACF functions
"""

import numpy as np
from old.start_byAtom_Function import get_EFG_data
from old.average_by_atom_Function import calculate_ACF, plot_ACF
import time

# # DME - NoAnion
# # path_Gromacs = "/home/santi/mendieta/DME_no-anion/"
# # path_MDrelax = "/home/santi/MD/MDRelax_results/test/"
# path_Gromacs = "C:/Users/Usuario/Documents/SantiM/MDdata/mendieta/DME_PS/"
# path_MDrelax = "C:/Users/Usuario/Documents/SantiM/MDdata/MDrelax_results/DME_PS/"

#######  Li+ - water
path_Gromacs = "C:/Users/Usuario/Documents/SantiM/MDdata/mendieta/Li-water/"
path_MDrelax = "C:/Users/Usuario/Documents/SantiM/MDdata/MDRelax_results/Li-water/"
cation_itp, anion_itp, solvent_itp = ["Li","none", "tip4p"] # as in .itp files
cation, anion, solvent = ["Li","none", "SOL"] # names
# salt = r"Li$^+$"
salt = r"Li$^+$"
Ncations = 1 # numero de Li+
runs_inds = range(6,7)
runs_prefix = "HQ"
mdp_prefix = "HQ.long"
runs_suffix = [f".{t*1000:.0f}_ps.long" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}.long" for t in runs_inds]

trajectory_format = ".xtc" # ".trr" or ".xtc"
topology_format = ".gro" # ".tpr" or ".gro"
forcefield = "Madrid.ff"

t0 = time.time()
# get_EFG_data(path_Gromacs, path_MDrelax,
#              species = [cation, anion, solvent],
#              species_itp = [cation_itp, anion_itp, solvent_itp],
#              Ncations = Ncations,
#              runs_prefix = runs_prefix,
#              runs_suffix = runs_suffix,
#              runs_suffix_gro = runs_suffix_gro,
#              trajectory_format = trajectory_format,
#              topology_format = topology_format,
#              forcefield=forcefield,
#              mdp_prefix=mdp_prefix
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")

#%%

t0 = time.time()
calculate_ACF(path_MDrelax,
              savepath = path_MDrelax,
              species = [cation, anion, solvent],              
              Ncations = Ncations,
              runs_prefix = runs_prefix,
              runs_suffix = runs_suffix)             
print(f"ACF time: {time.time()-t0:.0f} s")


t0 = time.time()
plot_ACF(path_MDrelax,
              savepath = path_MDrelax,
              species = [cation, anion, solvent],              
              salt = salt,
              Ncations = Ncations,
              runs_prefix = runs_prefix,
              runs_suffix = runs_suffix)             
print(f"plots time: {time.time()-t0:.0f} s")
# %%