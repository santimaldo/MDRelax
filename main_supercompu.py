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


forcefield = "park.ff"
# # DME - small-boxes NoAnion
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\DME_small-boxes\DME_no-anion/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\DME_small-boxes\DME_no-anion/"
# cation_itp, anion_itp, solvent_itp = ["Li","none", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","none", "DME"] # names
# salt = r"Li$^+$"
# Ncations = 1 # numero de Li+
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"


#######  Li+ - water
# path_Gromacs = "C:/Users/Usuario/Documents/SantiM/MDdata/mendieta/Li-water/"
# path_MDrelax = "C:/Users/Usuario/Documents/SantiM/MDdata/MDRelax_results/Li-water/"
# cation_itp, anion_itp, solvent_itp = ["Li","none", "tip4p"] # as in .itp files
# cation, anion, solvent = ["Li","none", "SOL"] # names
# # salt = r"Li$^+$"
# salt = r"Li$^+$"
# Ncations = 1 # numero de Li+
# runs_inds = range(7,11)
# runs_prefix = "HQ"
# mdp_prefix = "HQ.long"
# runs_suffix = [f".{t*1000:.0f}_ps.long" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}.long" for t in runs_inds]

# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".gro" # ".tpr" or ".gro"
# forcefield = "Madrid.ff"



# ##################################################
# # DME - small-boxes LiTFSI
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\DME\run_1ns/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\DME\run_1ns/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_DME", "DME_7CB8A2"] # as in .itp files
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\DME_small-boxes\DME_LiTFSI_NChR/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\DME_small-boxes\DME_LiTFSI_NChR/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS", "DME_7CB8A2"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "DME"] # names
# salt = r"LiTFSI"
# Ncations = 2 # numero de Li+
# runs_inds = range(6,7)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"

# t0 = time.time()
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
#             #  rcut=0.9 # nm
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


# t0 = time.time()
# calculate_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix)
#               #method='manual')             
# print(f"ACF time: {time.time()-t0:.0f} s")

#%%
# t0 = time.time()
# plot_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               salt = salt,
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix,
#               max_tau = 150)             
# print(f"plots time: {time.time()-t0:.0f} s")
# #%%
# # ##################################################
# ## TEGDME - small-boxes LiTFSI
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\TEGDME\run_10ns/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\TEGDME\run_10ns/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_TEGDME", "tegdme"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "TGD"] # names
# salt = r"LiTFSI"
# Ncations = 2 # numero de Li+
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"

# t0 = time.time()
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
#             #  rcut=0.9 # nm
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


# t0 = time.time()
# calculate_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix)
#               #method='manual')             
# print(f"ACF time: {time.time()-t0:.0f} s")
# #%%
# t0 = time.time()
# plot_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               salt = salt,
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix,
#               max_tau = 10000,
#               fignum=2)             
# print(f"plots time: {time.time()-t0:.0f} s")

#%%
# # ##################################################
# ## ACN - small-boxes LiTFSI
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\ACN\run_1ns/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\ACN\run_1ns/"
# cation_itp, anion_itp, solvent_itp = ["Li","TFS_ACN", "ACN"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "ACN"] # names
# salt = r"LiTFSI"
# Ncations = 2 # numero de Li+
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"

# t0 = time.time()
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
#             #  rcut=0.9 # nm
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


# t0 = time.time()
# calculate_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix)
#               #method='manual')             
# print(f"ACF time: {time.time()-t0:.0f} s")

# t0 = time.time()
# plot_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               salt = salt,
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix,
#               max_tau = 1000,
#               fignum=3)             
# print(f"plots time: {time.time()-t0:.0f} s")

# #%%
# # # ##################################################
# ## DOL - small-boxes LiTFSI
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\DOL\run_1ns/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\DOL\run_1ns/"
# cation_itp, anion_itp, solvent_itp = ["Li_DOL","TFS_DOL", "DOL"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "DOL"] # names
# salt = r"LiTFSI"
# Ncations = 2 # numero de Li+
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"

# t0 = time.time()
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
#             #  rcut=0.9 # nm
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


# t0 = time.time()
# calculate_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix)
#               #method='manual')             
# print(f"ACF time: {time.time()-t0:.0f} s")

# #%%
# t0 = time.time()
# plot_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               salt = salt,
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix,
#               max_tau = 100,
#               fignum=4)             
# print(f"plots time: {time.time()-t0:.0f} s")

# #
# # # ##################################################
# ## Diglyme - small-boxes LiTFSI
# path_Gromacs = r"C:\Users\Usuario\Documents\SantiM\MDdata\mendieta\LiTFSI_small-boxes\Diglyme\run_1ns/"
# path_MDrelax = r"C:\Users\Usuario\Documents\SantiM\MDdata\MDrelax_results\LiTFSI_small-boxes\Diglyme\run_1ns/"
# cation_itp, anion_itp, solvent_itp = ["Li_Diglyme","TFS_Diglyme", "Diglyme"] # as in .itp files
# cation, anion, solvent = ["Li","TFS", "DIG"] # names
# salt = r"LiTFSI"
# Ncations = 2 # numero de Li+
# runs_inds = range(6,11)
# mdp_prefix = "HQ"
# runs_prefix = "HQ"
# runs_suffix = [f".{t*1000:.0f}_ps" for t in runs_inds]
# runs_suffix_gro = [f".{t:.0f}" for t in runs_inds]
# trajectory_format = ".xtc" # ".trr" or ".xtc"
# topology_format = ".tpr" # ".tpr" or ".gro"

# t0 = time.time()
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
#             #  rcut=0.9 # nm
#              )
# print(f"EFG time: {time.time()-t0:.0f} s")


# t0 = time.time()
# calculate_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix)
#               #method='manual')             
# print(f"ACF time: {time.time()-t0:.0f} s")


# t0 = time.time()
# plot_ACF(path_MDrelax,
#               savepath = path_MDrelax,
#               species = [cation, anion, solvent],              
#               salt = salt,
#               Ncations = Ncations,
#               runs_prefix = runs_prefix,
#               runs_suffix = runs_suffix,
#               max_tau = 1000,
#               fignum=5)             
# print(f"plots time: {time.time()-t0:.0f} s")
# # %%
