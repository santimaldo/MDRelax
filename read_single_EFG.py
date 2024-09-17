#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:00:58 2023

@author: santi

read ACF functions and average
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid, simpson
import time
from Functions import *
plt.rcParams.update({'font.size': 12})


# DME - No anion
path_MDrelax = "/media/santi/Home/supercompu/MDrelax_results/DME_no-anion_bigbox/"
EFG_file = "EFG_total_HQ.6000_ps.dat"

filename = f"{path_MDrelax}{EFG_file}"
print("reading ", filename)
data = np.loadtxt(filename)
# read the time only once:
# data columns order:    
# t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
# 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
nn=0
# plt.plot(data[:,0], data[:,nn+1])            
Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                            
# this EFG_nn is (3,3,Ntimes) shaped
# nn is the cation index
# EFG_nn = np.array([[Vxx, Vxy, Vxz],
#                     [Vxy, Vyy, Vyz],
#                     [Vxz, Vyz, Vzz]])  
t1 = data[:,0]
Tr1 = Vxx + Vyy + Vzz


#---------------------
path_MDrelax = "/home/santi/MD/MDRelax_results/DME_PS/"
EFG_file = "EFG_total_HQ.6000_ps.dat"
# path_MDrelax = "/media/santi/Home/supercompu/MDrelax_results/DME_no-anion_bigbox/"
# EFG_file = "EFG_Li_HQ.6000_ps.dat"


filename = f"{path_MDrelax}{EFG_file}"
print("reading ", filename)
data = np.loadtxt(filename)
nn=0
# plt.plot(data[:,0], data[:,nn+1])            
Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                            
t2 = data[:,0]
Tr2 = Vxx + Vyy + Vzz
#%%


plt.figure(1)
plt.plot(t1, Tr1)
plt.plot(t2, Tr2)

plt.figure(2)
plt.plot(t2, Tr2)
# %%
