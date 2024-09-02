#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:05:03 2023

@author: santi

Creacion de una trayectoria (MRU) de prueba para testear el cofigo de EFG
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd



def Write_file(t, Li_pos, Cl_pos, box=[20.0,20.0,20.0], path='', step=0):
    # paso t a ps
    t = t/1000
    # paso a nanometros
    Lix, Liy, Liz = Li_pos/10
    Clx, Cly, Clz = Cl_pos/10
    
    # si las pos son negativas llevan 2 espacios. si son positivas, 3
    Lixsp, Liysp, Lizsp = [' '*int(2+((np.sign(Li_pos[kk])+1)!=0)*1) for kk
                          in range(3)]
    Clxsp, Clysp, Clzsp = [' '*int(2+((np.sign(Cl_pos[kk])+1)!=0)*1) for kk
                          in range(3)]
    
    box_x, box_y, box_z = np.array(box)/10
    filename = f"{path}LiCl_{t*1000:.0f}fs.gro"
    
    with open(filename, 'w') as f:
        f.write(f"1 LiCl molec, t= {t:.5f} step= {step:.0f}\n")
        f.write(f"2\n")
        f.write(f"    1LIT    Li     1{Lixsp}{Lix:.3f}{Liysp}{Liy:.3f}{Lizsp}{Liz:.3f}  1\n")
        f.write(f"    2CLO    Cl     1{Clxsp}{Clx:.3f}{Clxsp}{Cly:.3f}{Clxsp}{Clz:.3f}  -1\n")
        f.write(f"{box_x:.5f}   {box_y:.5f}   {box_z:.5f}")
        
#%%%
        
# path = "./caso_1/"
# # posiciones iniciales en Ang
# Li0 = np.array([10,10,0])
# Cl0 = np.array([10,10,1])
# # velocidades en Ang/fs
# v_Li = np.array([0, 0.0, 0.0]) 
# v_Cl = np.array([0, 0.0, 0.1]) 

path = "./caso_2/"
# posiciones iniciales en Ang
Li0 = np.array([15,0,0])
Cl0 = np.array([16,0,0])
# velocidades en Ang/fs
v_Li = np.array([-0.1, 0.0, 0.0]) 
v_Cl = np.array([ 0.0, 0.0, 0.0]) 

# path = "./caso_3/"
# # posiciones iniciales en Ang
# Li0 = np.array([0,0,0])
# Cl0 = np.array([1,1,1])/np.sqrt(3)
# # velocidades en Ang/fs
# v_Li = np.array([0.0, 0.0, 0.0]) 
# v_Cl = np.array([0.1, 0.1, 0.1])/np.sqrt(3) 


times = np.arange(0,101,10)

icount = 0
for t in times:
    # posiciones en t (Ang)
    Li_pos = Li0 + v_Li*t
    Cl_pos = Cl0 + v_Cl*t

    Write_file(t, Li_pos, Cl_pos, path=path, step=icount)
    icount+=1