#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:05:46 2023

@author: santi

Calculo T1
"""



e = 1.60217663 * 1e-19 # Coulomb
hbar = 1.054571817 * 1e-34 #joule seconds
ke = 8.9875517923 * 1e9# Vm/C, Coulomb constant

Q = -4.01 * (1e-15)**2 # m**2
I = 1.5 # spin 3/2


gamma = 0.17 # Sternhemmer factor 

tau_c = 6* 1e-12 # s
efg_variance = 0.020658403595027623 * ke**2 * e**2 / (1e-10)**6 # (V/m)^2


CQ = (2*I+3)*(e*Q/hbar)**2 / (I**2 * (2*I-1)) * (1/20)

R1 = CQ * (1+gamma)**2 * efg_variance * tau_c

T1 = 1/R1

print(T1)