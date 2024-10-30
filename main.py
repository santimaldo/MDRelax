#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  10 2024

@author: santi

read MD data and calculate ACF functions
"""



import numpy as np
from Functions import get_EFG_data, calculate_ACF, plot_ACF
import time

forcefield = "park.ff"

##### DME - no anion
# path_Gromacs = "/home/santi/mendieta/DME_no-anion_bigbox/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/tmp/"
# # path_MDrelax = "/home/santi/MD/MDRelax_results/DME_no-anion_bigbox_UHQ/upto5ps/"
# cation_itp, anion_itp, solvent_itp = ["Li","none", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","none", "DME"] # names
# salt = r"Li$^+$"
# Ncations = 20 # numero de Li+
# runs_inds = range(6,7)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".trr" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"
##### DME - no anion - SMALL BOX
path_Gromacs = "/home/santi/mendieta/DME_small-boxes/DME_no-anion/"
path_MDrelax = "/home/santi/MD/MDRelax_results/DME_small-boxes/DME_no-anion/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/DME_no-anion_bigbox_UHQ/upto5ps/"
cation_itp, anion_itp, solvent_itp = ["Li","none", "DME_7CB8A2"] # as in .itp files
cation, anion, solvent = ["Li","none", "DME"] # names
salt = r"Li$^+$"
Ncations = 1 # numero de Li+
runs_inds = range(6,11)
mdp_prefix = "HQ"
runs_prefix = "HQ"
runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
trajectory_format = ".trr" # ".trr" or ".xtc"
topology_format = ".tpr" # ".tpr" or ".gro"



######  DME-PS / TEGDME-PS
# path_Gromacs = "/home/santi/mendieta/DME_PS/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/DME_PS/"
# cation_itp, anion_itp, solvent_itp = ["Li","S6", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","S6", "DME"] # names

# path_Gromacs = "/home/santi/mendieta/TEGDME_PS/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/TEGDME_PS/"
# cation_itp, anion_itp, solvent_itp = ["Li","S6", "tegdme"] # as in .itp files
# cation, anion, solvent = ["Li","S6", "TGD"] # names
# # salt = r"Li$^+$"
# salt = r"Li$_2$S$_6$"
# Ncations = 40 # numero de Li+

# runs_inds = range(6,11)
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.Li2S6" for t in runs_inds]
# trajectory_format = ".trr" # ".trr" or ".xtc"
# topology_format = ".gro" # ".tpr" or ".gro"

#######  DME-LiTFSI
# path_Gromacs = "/home/santi/mendieta/DME_LiTFSI/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/DME_LiTFSI/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_DME", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "DME"] # names
# salt = r"LiTFSI"
# Ncations = 40 # numero de Li+
# runs_inds = range(6,11)
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.LiTFSI" for t in runs_inds]
# trajectory_format = ".trr" # ".trr" or ".xtc"
# topology_format = ".gro" # ".tpr" or ".gro"


# #######  TEGDME-LiTFSI
# path_Gromacs = "/home/santi/mendieta/TEGDME_LiTFSI/"
# path_MDrelax = "/home/santi/MD/MDRelax_results/TEGDME_LiTFSI/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_TEGDME", "tegdme"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "TGD"] # names
# # salt = r"Li$^+$"
# salt = r"Li$_2$S$_6$"
# Ncations = 40 # numero de Li+
# runs_inds = range(6,11)
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.LiTFSI" for t in runs_inds]

# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".gro" # ".tpr" or ".gro"


#######  Li+ - water
path_Gromacs = "/home/santi/mendieta/Li-water/"
cation_itp, anion_itp, solvent_itp = ["Li","none", "tip4p"] # as in .itp files
cation, anion, solvent = ["Li","none", "SOL"] # names
salt = r"Li$^+$"
Ncations = 1 # numero de Li+
runs_prefix = "HQ"
# - - - -
# path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/long/"
# runs_inds = range(6,11)
# mdp_prefix = "HQ.long"
# runs_suffix = [f".{t*1000:.0f}_ps.long" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.long" for t in runs_inds]
# - - - -
# path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/"
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# - - - -
# path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/freq0.1/"
# runs_inds = range(6,11)
# mdp_prefix = "HQ.long.freq0.1"
# runs_suffix = [f".{t*1000:.0f}_ps.long.freq0.1" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.long.freq0.1" for t in runs_inds]
# - - - -
path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/short/solo_Hidrogeno/"
runs_inds = range(6,11)
mdp_prefix = "HQ.long.skip10.upto100ps"
runs_suffix = [f".{t*1000:.0f}_ps.long.skip10.upto100ps" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}.long.skip10.upto100ps" for t in runs_inds]
# - - - -
trajectory_format = ".xtc" # ".trr" or ".xtc"
topology_format = ".tpr" # ".tpr" or ".gro"
forcefield = "Madrid.ff"


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
             )
print(f"EFG time: {time.time()-t0:.0f} s")


t0 = time.time()
calculate_ACF(path_MDrelax,
              savepath = path_MDrelax,
              species = [cation, anion, solvent],              
              Ncations = Ncations,
              runs_prefix = runs_prefix,
              runs_suffix = runs_suffix,
              method='scipy')             
print(f"ACF time: {time.time()-t0:.0f} s")
#%%
t0 = time.time()
plot_ACF(path_MDrelax,
            savepath = path_MDrelax,
            species = [cation, anion, solvent],              
            salt = salt,
            Ncations = Ncations,
            runs_prefix = runs_prefix,
            runs_suffix = runs_suffix,
            max_tau =  10)             
print(f"plots time: {time.time()-t0:.0f} s")
# %%
