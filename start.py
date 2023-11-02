#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:33:25 2023

@author: santi

First script to test the EFG summation
"""

import numpy as np
import matplotlib.pyplot as plt


# definir distancia y r!!!

r = np.array([3, 4, 12])
dist = np.linalg.norm(r)


EFG = np.zeros((3, 3))


for ii in range(3):
    for jj in range(3):
        Vij = 0
        q = -1
        ri = r[ii]
        rj = r[jj]
        Vij = 3*ri*rj/dist**5
        if ii == jj:
            Vij = Vij - 1/(dist**3)
        Vij = q*Vij

        EFG[ii, jj] = Vij

print(EFG*1e4)
