#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  10 12024

@author: santi

read MD data and calculate ACF functions
"""



import numpy as np
from start_byAtom_Function import get_EFG_data
from average_by_atom_Function import calculate_ACF, plot_ACF
import time

# DME - NoAnion
path_Gromacs = "/home/santi/mendieta/DME/"
path_MDrelax = "/home/santi/MD/MDRelax_results/DME_PS/"
# path_Gromacs = "C:/Users/Usuario/Documents/SantiM/MDdata/mendieta/DME/"
# path_MDrelax = "C:/Users/Usuario/Documents/SantiM/MDdata/MDrelax_results/DME_PS/"

cation_itp, anion_itp, solvent_itp = ["Li","S6", "DME_7CB8A2"] # as in .itp files
cation, anion, solvent = ["Li","S6", "DME"] # names
# salt = r"Li$^+$"
salt = r"Li$_2$S$_6$"
Ncations = 4 # numero de Li+

runs_inds = range(6,11)
runs_prefix = "HQ"
runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]



trajectory_format = ".trr" # ".trr" or ".xtc"
topology_format = ".gro" # ".tpr" or ".gro"


# t0 = time.time()
# get_EFG_data(path_Gromacs, path_MDrelax,
#              species = [cation, anion, solvent],
#              species_itp = [cation_itp, anion_itp, solvent_itp],
#              Ncations = Ncations,
#              runs_prefix = runs_prefix,
#              runs_suffix = runs_suffix,
#              runs_suffix_gro = runs_suffix_gro,
#              trajectory_format = trajectory_format,
#              topology_format = topology_format
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


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
print(f"ACF time: {time.time()-t0:.0f} s")