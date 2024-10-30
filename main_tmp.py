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

local = "/home/santi" # Sriracha
# local = "T:/"

forcefield = "park.ff"
#######  Li+ - water
path_Gromacs = f"{local}/mendieta/Li-water/"
cation_itp, anion_itp, solvent_itp = ["Li","none", "tip4p"] # as in .itp files
cation, anion, solvent = ["Li","none", "SOL"] # names
salt = r"Li$^+$"
Ncations = 1 # numero de Li+
runs_prefix = "HQ"

path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/short/solo_Oxigeno/Rescaling_q/"
runs_inds = range(6)
mdp_prefix = "HQ.long.skip10.upto100ps"
runs_suffix = [f".{t*1000:.0f}_ps.long.skip10.upto100ps" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}.long.skip10.upto100ps" for t in runs_inds]
# - - - -
trajectory_format = ".xtc" # ".trr" or ".xtc"
topology_format = ".tpr" # ".tpr" or ".gro"
forcefield = "Madrid.ff"


for factor_q in [1,0.5,0.1,0.9,0.75]:
    print("=+="*50)
    print(f"rescaling charge by a factor of {factor_q}")
    t0 = time.time()
    get_EFG_data(path_Gromacs, path_MDrelax+f"factor-q_{factor_q:.2f}",
                species = [cation, anion, solvent],
                species_itp = [cation_itp, anion_itp, solvent_itp],
                Ncations = Ncations,
                runs_prefix = runs_prefix,
                runs_suffix = runs_suffix,
                runs_suffix_gro = runs_suffix_gro,
                trajectory_format = trajectory_format,
                topology_format = topology_format,
                forcefield=forcefield,
                mdp_prefix=mdp_prefix,
                factor_q = factor_q
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
