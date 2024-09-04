import matplotlib.pyplot as plt
import numpy as np


path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/MDRelax/"
runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
solvent = "DME"

ACFs = []
for run in runs:
    
    data = np.loadtxt(f"{path}/ACF_{run}.dat")
    ACFs 