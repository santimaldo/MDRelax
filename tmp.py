import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

#----------------------------------------------------------
#caja original
fig, axs = plt.subplots(num=1, ncols=1, nrows=3, figsize=(12,12))
for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlim([-20,60])
    ax.set_ylim([-20,60])
    L = box[0]
    ax.add_patch(Rectangle((0, 0), L, L, fc='none', ec='blue'))
 
# xy
axs[0].scatter(group.positions[:, 0], group.positions[:, 1], marker='x')
circle_center = (cation_in_group.position[0], cation_in_group.position[1])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[0].add_artist(c)
axs[0].set_title("x-y")
# yz
axs[1].scatter(group.positions[:, 1], group.positions[:, 2], marker='x')
circle_center = (cation_in_group.position[1], cation_in_group.position[2])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[1].add_artist(c)
axs[1].set_title("y-z")
# xz
axs[2].scatter(group.positions[:, 0], group.positions[:, 2], marker='x')
circle_center = (cation_in_group.position[0], cation_in_group.position[2])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[2].add_artist(c)
axs[2].set_title("x-z")


#----------------------------------------------------------
# centrando un litio y aplicando pbc
fig, axs = plt.subplots(num=2, ncols=1, nrows=3, figsize=(12,12))
for ax in axs:
    ax.set_aspect('equal')
    ax.set_xlim([-20,60])
    ax.set_ylim([-20,60])
    L = box[0]
    ax.add_patch(Rectangle((0, 0), L, L, fc='none', ec='orange'))
 
# xy
axs[0].scatter(group_newpositions_pbc[:, 0], group_newpositions_pbc[:, 1], marker='x', color='orange')
circle_center = (center[0], center[1])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[0].add_artist(c)
axs[0].set_title("x-y")
# yz
axs[1].scatter(group_newpositions_pbc[:, 1], group_newpositions_pbc[:, 2], marker='x', color='orange')
circle_center = (center[1], center[2])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[1].add_artist(c)
axs[1].set_title("y-z")
# xz
axs[2].scatter(group_newpositions_pbc[:, 0], group_newpositions_pbc[:, 2], marker='x', color='orange')
circle_center = (center[0], center[2])
c=plt.Circle((circle_center), radius=1, color='red', alpha=.5)
axs[2].add_artist(c)
axs[2].set_title("x-z")