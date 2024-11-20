"""
Calculo las rfs de distintos pares:
Li-S
St-St inter e  intramolecular

(descomentar el de interes)
"""
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import pandas as pd
from Functions import *

# #%%==============================================
# # g(r) Li-O rdf (Li+ in water)
# # GROMACS file
# path_Gromacs = "/home/santi/mendieta/CHARMM/TEGDME/nmolec_100/"
# file = "HQ.6"
# terminal_1 = "H13 H12 H16"
# terminal_2 = "H1 H2 H3"

# u = mda.Universe(f"{path_Gromacs}{file}.tpr", f"{path_Gromacs}{file}.xtc")

# #%%
# group1 = u.select_atoms(f"name {terminal_1}")
# group2 = u.select_atoms(f"name {terminal_2}")    
# rdf = InterRDF(group1, group2)
# rdf.run()    
# bins = rdf.bins[1:]
# rdf = rdf.rdf[1:]

# plt.figure(1)
# plt.plot(bins, rdf, '-')
# plt.xlabel(r"$H_{terminal}-H_{terminal}$ [$10^{-10}$ m]")
# plt.ylabel(r"$g(r)$")
# plt.legend()
#%% INTRA=====================================

# GROMACS file
path_Gromacs = "/home/santi/mendieta/CHARMM/TEGDME/nmolec_100/"
file = "HQ.6"
terminal_1 = "H13 H12 H16"
terminal_2 = "H1 H2 H3"

# Load the Universe
u = mda.Universe(f"{path_Gromacs}{file}.tpr", f"{path_Gromacs}{file}.xtc")

# Select atoms for terminal groups
group1 = u.select_atoms(f"name {terminal_1}")
group2 = u.select_atoms(f"name {terminal_2}")

# Ensure we calculate intramolecular RDF only
rdf_intra = None
bins = None

# Iterate over molecules
for molecule in u.residues:
    mol_group1 = molecule.atoms.select_atoms(f"name {terminal_1}")
    mol_group2 = molecule.atoms.select_atoms(f"name {terminal_2}")
    
    # Skip if one of the groups is empty
    if len(mol_group1) == 0 or len(mol_group2) == 0:
        continue

    # Compute the RDF for this molecule
    rdf = InterRDF(mol_group1, mol_group2)
    rdf.run()

    # Accumulate the RDFs
    if rdf_intra is None:
        rdf_intra = rdf.rdf
        bins = rdf.bins
    else:
        rdf_intra += rdf.rdf

# Normalize the RDF to average over all molecules
rdf_intra /= len(u.residues)

# Plot the RDF
bins = bins[1:]  # Skip the first bin (r = 0)
rdf_intra = rdf_intra[1:]

bins_ch = bins
rdf_ch = rdf_intra

print("CHARMM listo v/")

#%%##################################
# GROMACS file
path_Gromacs = "/home/santi/mendieta/TEGDME_small-boxes/TEGDME_LiTFSI/run_1ns/"
file = "HQ.6"
terminal_1 = "H0S H0U H0T"
terminal_2 = "H0Z H10 H11"

# Load the Universe
u = mda.Universe(f"{path_Gromacs}{file}.tpr", f"{path_Gromacs}{file}.xtc")

# Select atoms for terminal groups
group1 = u.select_atoms(f"name {terminal_1}")
group2 = u.select_atoms(f"name {terminal_2}")

# Ensure we calculate intramolecular RDF only
rdf_intra = None
bins = None

# Iterate over molecules
for molecule in u.residues:
    mol_group1 = molecule.atoms.select_atoms(f"name {terminal_1}")
    mol_group2 = molecule.atoms.select_atoms(f"name {terminal_2}")
    
    # Skip if one of the groups is empty
    if len(mol_group1) == 0 or len(mol_group2) == 0:
        continue

    # Compute the RDF for this molecule
    rdf = InterRDF(mol_group1, mol_group2)
    rdf.run()

    # Accumulate the RDFs
    if rdf_intra is None:
        rdf_intra = rdf.rdf
        bins = rdf.bins
    else:
        rdf_intra += rdf.rdf

# Normalize the RDF to average over all molecules
rdf_intra /= len(u.residues)

# Plot the RDF
bins = bins[1:]  # Skip the first bin (r = 0)
rdf_intra = rdf_intra[1:]

bins_op = bins
rdf_op = rdf_intra


#%%
plt.figure(1)
plt.plot(bins_op, rdf_op, '-', label="OPLS")
plt.plot(bins_ch, rdf_ch, '-', label="CHARMM36")
plt.xlabel(r"$H_{terminal}-H_{terminal}$ [$10^{-10}$ m]")
plt.ylabel(r"$g(r)$")
plt.title("TEGDME: Intramolecular RDF")
plt.legend()
plt.show()
