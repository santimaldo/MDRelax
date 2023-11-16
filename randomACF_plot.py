#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:40:19 2023

@author: santi
"""

import numpy as np
import matplotlib.pyplot as plt




names = ["/home/santi/MD/testsfiles/VariableRandom/RandomACF_Nsteps1000.dat",
         "/home/santi/MD/testsfiles/VariableRandom/RandomACF_Nsteps5000.dat",
         "/home/santi/MD/testsfiles/VariableRandom/RandomACF_Nsteps10000.dat",
         "/home/santi/MD/testsfiles/VariableRandom/RandomACF_dtCorr100_Nsteps50000.dat"]

Nsteps = [1000,5000,10000,50000]



names = ["/home/santi/MD/testsfiles/VariableRandom/RandomACF_dtCorr500_Nsteps50000.dat",
         "/home/santi/MD/testsfiles/VariableRandom/RandomACF_dtCorr100_Nsteps50000.dat"]


for ii in range(len(names)):
    filename = names[ii]
    datos = np.loadtxt(filename)
    
    tau = datos[:,0]
    acf = datos[:,1]
    
    plt.figure(2)
    plt.plot(tau/1000, acf/acf[0], 'o-')#), label=f"Nsteps: {Nsteps[ii]}")
    # plt.plot(tau, 1.25*np.exp(-tau/tau_c), 'k--')
    plt.xlim([0,tau[-1]/2/1000])

plt.xlabel(r"$\tau$ [ps]")
plt.ylabel("ACF")
plt.legend()
plt.show()
    