#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 20  2024

@author: santi

First script to test the 1H relaxation summation
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rdf
import pandas as pd
import nmrformd as nmrmd
import time
from scipy.integrate import cumulative_trapezoid

path = "../DATA/2023-12_DME/500ps/frames_HQ_2/"
savepath = "../DATA/2023-12_DME/results/1H/"
u = mda.Universe(path+"HQ_npt-500ps_2.tpr", path+"HQ_npt-500ps_2.xtc")

#path = "../DATA/2023-12_TEGDME/500ps/frames_HQ_1/"
#u = mda.Universe(path+"HQ_npt-500ps_1.tpr", path+"HQ_npt-500ps_1.xtc")

dt = 0.01 # ps
ni = 40 # "number_i"
start_time = time.time()
end_time = time.time()
box = u.dimensions


H_group = u.select_atoms("name H*")
Li_group = u.select_atoms("name Li*")

## selecciono los atomos de H de la "primera esfera"------------------
# Define a distance threshold
distance_threshold = 3.5
# Compute the distance array
distances = distance_array(H_group.positions, Li_group.positions)
# Create a mask for atoms within the distance threshold
within_threshold = distances < distance_threshold
# Determine which H atoms are within the threshold
H_group_within_threshold = H_group[np.any(within_threshold, axis=1)]
# Create a new AtomGroup called H_bond with the selected atoms
H_bond = u.select_atoms('index ' + ' '.join(map(str, H_group_within_threshold.indices)))
print("Number of selected H_bond atoms:", len(H_bond))
#---------------------------------------------------------------------
H_free = H_group.difference(H_bond)


print("calculando intra...")
nmr_H_group_intra = nmrmd.NMR(u, atom_group=H_group, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="intra_molecular")

elapsed_time = time.time() - end_time
end_time = time.time()
print(f'Time elapsed: {elapsed_time/60} minutes')
print("calculando inter...")
nmr_H_group_inter = nmrmd.NMR(u, atom_group=H_group, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="inter_molecular")
elapsed_time = time.time() - end_time
end_time = time.time()
print(f'Time elapsed: {elapsed_time/60} minutes')

print("calculando total...")
nmr_H_group = nmrmd.NMR(u, atom_group=H_group, isotropic=True, actual_dt=dt, number_i=ni)
elapsed_time = time.time() - end_time
end_time = time.time()
print(f'Time elapsed: {elapsed_time/60} minutes')



R1dispersion_intra = nmr_H_group_intra.R1
R1dispersion_inter = nmr_H_group_inter.R1
R1dispersion = nmr_H_group.R1

R2dispersion_intra = nmr_H_group_intra.R2
R2dispersion_inter = nmr_H_group_inter.R2
R2dispersion = nmr_H_group.R2

frequency = nmr_H_group.f

ACF_intra = nmr_H_group_intra.gij[0,:]
ACF_inter = nmr_H_group_inter.gij[0,:]
ACF = nmr_H_group.gij[0,:]

tau = np.arange(ACF.size)*dt

data = np.array([tau, ACF, ACF_intra, ACF_inter]).T
#%%
if ni!=0:
    header = f"tau (ps)    ACF    ACF_intra    ACF_inter \n "\
             f"calculated with {ni} atoms (over {H_group.n_atoms} total H atoms)"
else:
    header = f"tau (ps)    ACF    ACF_intra    ACF_inter \n "\
             f"calculated all H atoms: {H_group.n_atoms}"
np.savetxt(savepath+f"ACF_AllH_number-i_{ni}.dat", data, header=header)






plt.figure(1)
plt.loglog(frequency, 1/R1dispersion_intra, 'o-', label="Intra-molecular")
plt.loglog(frequency, 1/R1dispersion_inter, 'o-', label="Inter-molecular")
plt.loglog(frequency, 1/R1dispersion, 'o-', label="Full")
plt.legend()
plt.xlabel("frequency [Hz]")
plt.ylabel("T1 [s]")


plt.figure(2)
plt.loglog(frequency, 1/R2dispersion_intra, 'o-', label="Intra-molecular")
plt.loglog(frequency, 1/R2dispersion_inter, 'o-', label="Inter-molecular")
plt.loglog(frequency, 1/R2dispersion, 'o-', label="Full")
plt.legend()
plt.xlabel("frequency [Hz]")
plt.ylabel("T2 [s]")

plt.figure(3)
plt.plot(np.arange(ACF_intra.size)*u.trajectory.dt, ACF_intra.T, 'o-', label="Intra-molecular")
plt.plot(np.arange(ACF_inter.size)*u.trajectory.dt, ACF_inter.T, 'o-', label="Inter-molecular")
plt.plot(np.arange(ACF.size)*u.trajectory.dt, ACF.T, 'o-', label="Full")
plt.legend()
plt.xlabel(r"$\tau$ [ps]")
plt.ylabel("ACF")
plt.show()

plt.figure(4)
plt.plot(tau, cumulative_trapezoid(ACF_intra[0,:], x=tau, initial=0), 'o-', label="Intra-molecular")
plt.plot(tau, cumulative_trapezoid(ACF_inter[0,:], x=tau, initial=0), 'o-', label="Inter-molecular")
plt.plot(tau, cumulative_trapezoid(ACF[0,:], x=tau, initial=0), 'o-', label="Full")
plt.legend()
plt.xlabel(r"$\tau$ [ps]")
plt.ylabel(r"$\int_0^{\tau}ACF d\tau'$")
plt.show()




# %%
