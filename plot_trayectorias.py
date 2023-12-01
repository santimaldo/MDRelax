#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:20:18 2023

@author: santi

correr solo las celdas, no con F5
"""


#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

lip = np.array(Li_positions)
li1 = lip[:,0,:]
li2 = lip[:,1,:]

stp = np.array(St_positions)
st1 = stp[:,0,0,:]
st2 = stp[:,0,1,:]

sip = np.array(Si_positions)[:,0,:,:]

# markers:
lim, stm, sim = 'o','x','x'
lim, stm, sim = '','',''


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(li1[:,0], li1[:,1], li1[:,2], marker = lim, label = r'$Li_1$')
ax.plot(li2[:,0], li2[:,1], li2[:,2], marker = lim, label = r'$Li_2$')

ax.plot(st1[:,0], st1[:,1], st1[:,2], marker = stm, color='goldenrod', 
        label = " S terminal")
ax.plot(st2[:,0], st2[:,1], st2[:,2], marker = stm, color='goldenrod')

for jj in range(4):
    if jj==0:    
        ax.plot(sip[:,jj,0], sip[:,jj,1], sip[:,jj,2], marker = sim, 
                color='gold', label = "S interior")
    else:
        ax.plot(sip[:,jj,0], sip[:,jj,1], sip[:,jj,2], marker = sim, 
                color='gold')

ax.set_xlabel("x [Ang]")
ax.set_ylabel("y [Ang]")
ax.set_zlabel("z [Ang]")
plt.legend()
# ax.scatter(*li1.T[0], color = 'red')
plt.show()


#%% Proyeccion x-y

i=1
j=2

fig, ax = plt.subplots(num=2)
ax.set_aspect('equal')
ax.plot(li1[:,i], li1[:,j], marker = lim, label = r'$Li_1$')
ax.plot(li2[:,i], li2[:,j], marker = lim, label = r'$Li_2$')

ax.plot(st1[:,i], st1[:,j], marker = stm, color='goldenrod', 
        label = " S terminal")
ax.plot(st2[:,i], st2[:,j], marker = stm, color='goldenrod')

for jj in range(4):
    if jj==0:    
        ax.plot(sip[:,jj,i], sip[:,jj,j], marker = sim, 
                color='gold', label = "S interior")
    else:
        ax.plot(sip[:,jj,i], sip[:,jj,j], marker = sim, 
                color='gold')

ax.set_xlabel("x [Ang]")
if j==1:
    ax.set_ylabel("y [Ang]")
if j==2:
    ax.set_ylabel("z [Ang]")
    if i==1:
        ax.set_xlabel("y [Ang]")

# plt.legend()
plt.show()