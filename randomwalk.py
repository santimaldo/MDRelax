#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:29:12 2023

@author: santi
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

# Function to apply periodic boundary conditions


def apply_periodic_Boundary(position, box_size):
    return position % box_size


def random_walk_step(positions, box_size, step_length):
    num_particles = positions.shape[0]
    # Perform random walk for num_steps
    random_step = np.random.uniform(-step_length,
                                    step_length, size=(num_particles, 3))
    newpositions = np.mod(positions + random_step, box_size)
    return newpositions


def export_gro_file(positions_A, positions, output_filename, box_size, time,
                    charge_A=1, charge_p=-0.5):
    """
    Export particle positions to a .gro file.

    Parameters:
        positions (numpy.ndarray): Array containing particle positions in shape (N, 3).
        output_filename (str): Name of the output .gro file.
        box_size (numpy.ndarray): Array containing the size of the simulation box in shape (3,).
    """
    num_A_particles = positions_A.shape[0]
    num_p_particles = positions.shape[0] 
    num_particles = num_A_particles + num_p_particles  
    # paso a nm
    positions_A = positions_A/10
    positions = positions/10
    box_size = box_size/10
    with open(output_filename, 'w') as gro_file:
        #"    1WATER  OW1    1   0.126   1.624   1.679  0.1227"
        gro_file.write(f"random walk, t= {time} step= {0.000}\n")
        gro_file.write(f"{num_particles}\n")        
        for i in range(num_A_particles):            
            resnum, resname, atom, partnum = 1,'AA', 'A', i+1
            gro_file.write(f"{resnum:5d}{resname:5s}{atom:5s}{partnum:5d}")
            for ii in range(3):
                gro_file.write(f"{positions_A[i,ii]:8.3f}")
            gro_file.write(f"{charge_A:8.4f}")
            gro_file.write("\n")

        for i in range(num_p_particles):            
            resnum, resname, atom, partnum = 2,'PP', 'p', i+1+num_A_particles
            gro_file.write(f"{resnum:5d}{resname:5s}{atom:5s}{partnum:5d}")
            for ii in range(3):
                gro_file.write(f"{positions[i,ii]:8.3f}")
            gro_file.write(f"{charge_p:8.4f}")
            gro_file.write("\n")
        gro_file.write(
            f"{box_size[0]:.5f}   {box_size[1]:.5f}   {box_size[2]:.5f}\n")


# %%
path = "../testsfiles/randomwalk/"

# Inputs
box_size = np.array([20, 20, 20])
step_length = 0.05
num_steps = 1000
num_p_particles = 100
num_A_particles = 1
dt = 0.1  # ps

initial_positions = np.random.rand(num_p_particles, 3) * box_size
# initial_positions_A = np.random.rand(num_A_particles, 3) * box_size
initial_positions_A = np.ones([num_A_particles, 3]) * box_size/2

# Simulate random walks
# particles = random_walk_3d(num_particles=101, box_size=box_size, step_length=step_length, num_steps=num_steps)
trajectories_A = np.zeros([num_A_particles, num_steps, 3])
positions_A = initial_positions_A
positions = initial_positions
for nn in range(num_steps):
    print(f"Step: {nn}")
    time = dt*nn
    trajectories_A[:, nn, :] = positions_A
    positions_A = random_walk_step(positions_A, box_size, step_length)
    positions = random_walk_step(positions, box_size, step_length)

    output_filename = f"{path}randomwalk_{time*1000:.0f}fs.gro"
    export_gro_file(positions_A, positions, output_filename, box_size, time)


# Plot the trajectory of particle B
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectory of Particle B')
for jj in range(num_A_particles):
    ax.plot(trajectories_A[jj, :, 0], trajectories_A[jj, :, 1],
            trajectories_A[jj, :, 2], marker='o', linestyle='')
plt.show()
