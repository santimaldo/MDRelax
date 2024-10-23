
"""
Calculo la evolucion en el tiempo de St-St intramolecular

(descomentar el de interes)
"""
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import pandas as pd
from Functions import *



# path_Gromacs = "/home/santi/mendieta/DME_small-boxes/DME/"
# u = mda.Universe(f"{path_Gromacs}HQ.6.tpr", f"{path_Gromacs}HQ.6.trr")
# cations = u.select_atoms(f"name Li*")
# anions = u.select_atoms(f"name S6*")

### Li-S
# rdf_S6 = InterRDF(cations, anions)
# rdf_S6t = InterRDF(cations, anions.select_atoms("name S6t"))
# rdf_S6i = InterRDF(cations, anions.select_atoms("name S6i"))
# rdf_S6.run()
# rdf_S6t.run()
# rdf_S6i.run()
# plt.plot(rdf_S6.bins, rdf_S6.rdf, 'o-', label="Li-S")
# plt.plot(rdf_S6t.bins, rdf_S6t.rdf, 'o-', label="Li-S(terminal)")
# plt.plot(rdf_S6i.bins, rdf_S6i.rdf, 'o-', label="Li-S(interior)")
# plt.legend()

# #%%==============================================
# # g(r) St-St INTERmolecular HQ
# # path_Gromacs = "/home/santi/mendieta/DME_PS/"
# path_Gromacs = "/home/santi/mendieta/DME_PS_0.5M/"
# # path_Gromacs = "/home/santi/mendieta/TEGDME_PS/"
# binss = []
# rdfs = []
# for ii in range(6,11):
#     # u = mda.Universe(f"{path_Gromacs}HQ.{ii}.Li2S6.gro", f"{path_Gromacs}HQ.{ii}.Li2S6.trr")
#     u = mda.Universe(f"{path_Gromacs}HQ.{ii}.tpr", f"{path_Gromacs}HQ.{ii}.trr")

#     cations = u.select_atoms(f"name Li*")
#     anions = u.select_atoms(f"name S6*")
#     anions_t = anions.select_atoms("name S6t")
#     rdf_S6t = InterRDF(anions_t, anions_t)
#     rdf_S6t.run()    
#     binss.append(rdf_S6t.bins[1:])
#     rdfs.append(rdf_S6t.rdf[1:])
# rdfs = np.array(rdfs)
# plt.figure(1)
# for ii in range(5):
#     plt.plot(binss[ii], rdfs[ii,:], 'o-', label=f"S6t-S6t, run={ii+6}")
# plt.plot(binss[0], np.mean(rdfs, axis=0), 'ko-', label=f"S6t-S6t, Mean over runs")
# plt.xlabel(r"$S_{6t}-S_{6t}$ [$10^{-10}$ m]")
# plt.ylabel(r"$g(r)$")
# plt.legend()

#%%==============================================
# g(r) St-St INTRAmolecular HQ
# path_Gromacs = "/home/santi/mendieta/DME_PS/"
path_Gromacs = "/home/santi/mendieta/DME_PS_0.5M/"
# path_Gromacs = "/home/santi/mendieta/TEGDME_PS/"
Nruns = 5
#### rdfs size: [runs, anions, bins]
rdfs_total = np.zeros([Nruns, 20, 74])
for jj in range(Nruns):
    ii = jj+6
    print(f"RUN: {ii}")
    # u = mda.Universe(f"{path_Gromacs}HQ.{ii}.Li2S6.gro", f"{path_Gromacs}HQ.{ii}.Li2S6.trr")
    u = mda.Universe(f"{path_Gromacs}HQ.{ii}.tpr", f"{path_Gromacs}HQ.{ii}.trr")    
    anions = u.select_atoms(f"name S6*")
    # recorro cada azufre:
    idx=-1
    S_resids = np.unique([atom.resid for atom in anions])
    for S_resid in S_resids:
        idx+=1
        anion_resid = anions.select_atoms(f"resid {S_resid}")
        anion_t = anion_resid.select_atoms("name S6t")
        rdf_S6t = InterRDF(anion_t, anion_t)
        rdf_S6t.run()
        rdfs_total[jj, idx, :] = rdf_S6t.rdf[1:]   
bins = rdf_S6t.bins[1:]
rdfs = np.mean(rdfs_total, axis=1)
plt.figure(1)
for ii in range(Nruns):
    plt.plot(bins, rdfs[ii,:], 'o-', label=f"S6t-S6t, run={ii+6}")
plt.plot(bins, np.mean(rdfs, axis=0), 'ko-', label=f"S6t-S6t, Mean over runs")
plt.xlabel(r"Intramolecular $S_{t}-S_{t}$ [$10^{-10}$ m]")
plt.ylabel(r"$g(r)$")
plt.legend()



# %%
