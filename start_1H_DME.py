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
box = u.dimensions


H_group = u.select_atoms("name H*")
Li_group = u.select_atoms("name Li*")

## selecciono los atomos de H de la "primera esfera"------------------
# Define a distance threshold
distance_threshold = 3.8
# Compute the distance array
distances = mda.lib.distances.distance_array(H_group, Li_group)
# Create a mask for atoms within the distance threshold
within_threshold = distances < distance_threshold
# Determine which H atoms are within the threshold
H_group_within_threshold = H_group[np.any(within_threshold, axis=1)]
# Create a new AtomGroup called H_bond with the selected atoms
H_bond = u.select_atoms('index ' + ' '.join(map(str, H_group_within_threshold.indices)))
print("Number of selected H_bond atoms:", len(H_bond))
#---------------------------------------------------------------------
H_free = H_group.difference(H_bond)
n_i = len(H_bond)

print("Comenzazmos con H_bond")
print("calculando intra...") #(si no es intra, es inter_molecular)
nmr_H_bond_intra = nmrmd.NMR(u, atom_group=H_bond, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="intra_molecular", neighbor_group=H_group)
print("calculando inter...")
nmr_H_bond_inter= nmrmd.NMR(u, atom_group=H_bond, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="inter_molecular", neighbor_group=H_group)
print("calculando total...")
nmr_H_bond= nmrmd.NMR(u, atom_group=H_bond, isotropic=True, actual_dt=dt, number_i=ni, neighbor_group=H_group)
elapsed_time = time.time() - end_time
end_time = time.time()
#---------------------------------------------------------------
print("Continuamos con H_free")
print("calculando intra...") #(si no es intra, es inter_molecular)
nmr_H_free_intra = nmrmd.NMR(u, atom_group=H_free, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="intra_molecular", neighbor_group=H_group)
print("calculando inter...")
nmr_H_free_inter = nmrmd.NMR(u, atom_group=H_free, isotropic=True, actual_dt=dt, number_i=ni,
                         type_analysis="inter_molecular", neighbor_group=H_group)
print("calculando total...")
nmr_H_free = nmrmd.NMR(u, atom_group=H_free, isotropic=True, actual_dt=dt, number_i=ni, neighbor_group=H_group)
elapsed_time = time.time() - end_time
end_time = time.time()
print(f'Time elapsed: {elapsed_time/60} minutes')


#%%
R1dispersion_bond_intra = nmr_H_bond_intra.R1
R1dispersion_bond_inter = nmr_H_bond_inter.R1
R1dispersion_bond = nmr_H_bond.R1
R1dispersion_free_intra = nmr_H_free_intra.R1
R1dispersion_free_inter = nmr_H_free_inter.R1
R1dispersion_free = nmr_H_free.R1


R2dispersion_bond_intra = nmr_H_bond_intra.R2
R2dispersion_bond_inter = nmr_H_bond_inter.R2
R2dispersion_bond = nmr_H_bond.R2
R2dispersion_free_intra = nmr_H_free_intra.R2
R2dispersion_free_inter = nmr_H_free_inter.R2
R2dispersion_free = nmr_H_free.R2
frequency_bond = nmr_H_bond.f
frequency_free = nmr_H_free.f

ACF_bond_intra = nmr_H_bond_intra.gij[0,:]
ACF_bond_inter = nmr_H_bond_inter.gij[0,:]
ACF_bond = nmr_H_bond.gij[0,:]
ACF_free_intra = nmr_H_free_intra.gij[0,:]
ACF_free_inter = nmr_H_free_inter.gij[0,:]
ACF_free = nmr_H_free.gij[0,:]

tau = np.arange(ACF_bond.size)*dt

# # data = np.array([tau, ACF, ACF_intra, ACF_inter]).T
# #%%
# if ni!=0:
#     header = f"tau (ps)    ACF    ACF_intra    ACF_inter \n "\
#              f"calculated with {ni} atoms (over {H_group.n_atoms} total H atoms)"
# else:
#     header = f"tau (ps)    ACF    ACF_intra    ACF_inter \n "\
#              f"calculated all H atoms: {H_group.n_atoms}"
# np.savetxt(savepath+f"ACF_AllH_number-i_{ni}.dat", data, header=header)




plt.figure(1)
plt.loglog(frequency_bond, 1/R1dispersion_bond_intra, 'o-', label="H_bond_intra")
plt.loglog(frequency_bond, 1/R1dispersion_bond_inter, 'o-', label="H_bond_inter")
plt.loglog(frequency_bond, 1/R1dispersion_bond, 'o-', label="H_bond")
plt.loglog(frequency_free, 1/R1dispersion_free_intra, 'o-', label="H_free_intra")
plt.loglog(frequency_free, 1/R1dispersion_free_inter, 'o-', label="H_free_inter")
plt.loglog(frequency_free, 1/R1dispersion_free, 'o-', label="H_free")
plt.legend()
plt.xlabel("frequency [Hz]")
plt.ylabel("T1 [s]")


plt.figure(2)
plt.loglog(frequency_bond, 1/R2dispersion_bond_intra, 'o-', label="H_bond_intra")
plt.loglog(frequency_bond, 1/R2dispersion_bond_inter, 'o-', label="H_bond_inter")
plt.loglog(frequency_bond, 1/R2dispersion_bond, 'o-', label="H_bond")
plt.loglog(frequency_free, 1/R2dispersion_free_intra, 'o-', label="H_free_intra")
plt.loglog(frequency_free, 1/R2dispersion_free_inter, 'o-', label="H_free_inter")
plt.loglog(frequency_free, 1/R2dispersion_free, 'o-', label="H_free")
plt.legend()
plt.xlabel("frequency [Hz]")
plt.ylabel("T2 [s]")

plt.figure(3)
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_intra.T, 'o-', label="H_bond_intra")
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_inter.T, 'o-', label="H_bond_inter")
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond.T, 'o-', label="H_bond")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_intra.T, 'o-', label="H_free_intra")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_inter.T, 'o-', label="H_free_inter")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free.T, 'o-', label="H_free")
plt.legend()
plt.xlabel(r"$\tau$ [ps]")
plt.ylabel("ACF")
plt.show()


plt.figure(4)
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_intra.T/ACF_bond_intra[0], 'o-', label="H_bond_intra")
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond_inter.T/ACF_bond_inter[0], 'o-', label="H_bond_inter")
plt.plot(np.arange(ACF_bond.size)*u.trajectory.dt, ACF_bond.T/ACF_bond[0], 'o-', label="H_bond")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_intra.T/ACF_free_intra[0], 'o-', label="H_free_intra")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free_inter.T/ACF_free_inter[0], 'o-', label="H_free_inter")
plt.plot(np.arange(ACF_free.size)*u.trajectory.dt, ACF_free.T/ACF_free[0], 'o-', label="H_free")
plt.legend()
plt.xlabel(r"$\tau$ [ps]")
plt.ylabel(r"$ACF_{NORM}$")
plt.show()

# plt.figure(4)
# plt.plot(tau, cumulative_trapezoid(ACF_intra[0,:], x=tau, initial=0), 'o-', label="Intra-molecular")
# plt.plot(tau, cumulative_trapezoid(ACF_inter[0,:], x=tau, initial=0), 'o-', label="Inter-molecular")
# plt.plot(tau, cumulative_trapezoid(ACF[0,:], x=tau, initial=0), 'o-', label="Full")
# plt.legend()
# plt.xlabel(r"$\tau$ [ps]")
# plt.ylabel(r"$\int_0^{\tau}ACF d\tau'$")
# plt.show()




# %%
