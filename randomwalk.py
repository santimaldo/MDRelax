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


def random_walk_step(positions, box_size, step_length, num_particles=1):
    # Perform random walk for num_steps
    random_step = np.random.uniform(-step_length,
                                    step_length, size=(num_particles, 3))
    newpositions = np.mod(positions + random_step, box_size)
    return newpositions


def export_gro_file(position_A, positions, output_filename, box_size, time,
                    charge_A=1, charge_p=-0.5):
    """
    Export particle positions to a .gro file.

    Parameters:
        positions (numpy.ndarray): Array containing particle positions in shape (N, 3).
        output_filename (str): Name of the output .gro file.
        box_size (numpy.ndarray): Array containing the size of the simulation box in shape (3,).
    """
    num_particles = positions.shape[0]
    with open(output_filename, 'w') as gro_file:
        gro_file.write(f"random walk, t={time} step={0.000}\n")
        gro_file.write(f"{num_particles}\n")
        gro_file.write(f"    1A   A   1   ")
        gro_file.write(" ".join(f"{coord:8.3f}" for coord in position_A[0]))
        gro_file.write(f"   {charge_A}\n")

        for i in range(num_particles):
            gro_file.write(f"    {(i+2)%4}p{i+1}   p   {i+2}   ")
            gro_file.write(" ".join("{:8.3f}".format(coord)
                           for coord in positions[i]))
            gro_file.write(f"   {charge_p}\n")
        gro_file.write(
            f"    {box_size[0]:.5f}   {box_size[1]:.5f}   {box_size[2]:.5f}\n")


# %%
# Inputs
box_size = np.array([20, 20, 20])
step_length = 0.05
num_steps = 100
num_particles = 100
dt = 0.01  # ps

initial_positions = np.random.rand(num_particles, 3) * box_size
initial_position_A = box_size/2

# Simulate random walks
# particles = random_walk_3d(num_particles=101, box_size=box_size, step_length=step_length, num_steps=num_steps)
trajectory_A = np.zeros([num_steps, 3])
position_A = initial_position_A
positions = initial_positions
for nn in range(num_steps):
    print(f"Step: {nn}")
    time = dt*nn
    trajectory_A[nn, :] = position_A
    position_A = random_walk_step(position_A, box_size, step_length)
    positions = random_walk_step(positions, box_size, step_length)

    output_filename = f"../testsfiles/randomwalk_{time*1000:.0f}fs.gro"
    export_gro_file(position_A, positions, output_filename, box_size, time)


# Plot the trajectory of particle B
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectory of Particle B')
ax.plot(trajectory_A[:, 0], trajectory_A[:, 1],
        trajectory_A[:, 2], marker='o', linestyle='-', color='b')
plt.show()
