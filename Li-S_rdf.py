


import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import pandas as pd
from Functions import *

path = "/home/santi/MD/GromacsFiles/2024-08-15_DME_2nd-test/"
species_list = ["Li", "S6", "DME_7CB8A2"]

runs = [f"{t:.1f}_ps" for t in [0.5,1,1.5,2]]
MDfiles = [f"HQ.{i}" for i in range(6,10)]


dt = 0.01 # ps (tiempo entre frames) ## parametro de la simulacion: automatizar.

Charges = get_Charges(species_list, path)

# loop over different runs
ii = 0    
rdfs = []
for idx in range(len(MDfiles)):    
    run = runs[idx]
    filename = MDfiles[idx]    
    # u = mda.Universe(f"{path}{filename}.tpr", f"{path}{filename}.trr")
    u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.trr")
    box=u.dimensions
    center = box[0:3]/2
    
    Li_resids = [atom.resid for atom in u.select_atoms("name Li*")]
    for Li_resid in Li_resids:
        ii+=1        
        rdf = InterRDF(u.select_atoms(f"resid {Li_resid}"), u.select_atoms("name S6t"))
        rdf.run()
        plt.figure(ii)
        plt.plot(rdf.bins, rdf.rdf, 'o-', label=f"run: {run}, Li resid {Li_resid}")
        plt.legend()
        plt.figure(ii*10)
        plt.plot(rdf.bins, np.gradient(rdf.rdf), 'o-', label=f"run: {run}, Li resid {Li_resid}")
        rdfs.append([rdf.bins, rdf.rdf])
        plt.legend()

plt.figure(0)
suma = 0
for rdf in rdfs:
    plt.plot(rdf[0], rdf[1], 'o-')
    suma+=rdf[1]
plt.figure(5454354354354)
plt.plot(rdfs[0][0], suma, 'o-')

