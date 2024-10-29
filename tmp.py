#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Functions import *
import time
from matplotlib.animation import FuncAnimation, PillowWriter


# #######  Li+ - water
path_Gromacs = "/home/santi/mendieta/Li-water/"
path_MDrelax = "/home/santi/MD/MDRelax_results/Li-water/long/"
cation_itp, anion_itp, solvent_itp = ["Li","none", "tip4p"] # as in .itp files
cation, anion, solvent = ["Li","none", "SOL"] # names
# salt = r"Li$^+$"
salt = r"Li$^+$"
Ncations = 1 # numero de Li+
runs_inds = range(7,8)
runs_prefix = "HQ"
mdp_prefix = "HQ.long"
runs_suffix = [f".{t*1000:.0f}_ps.long" for t in runs_inds]
runs_suffix_gro = [f".{t:.0f}.long" for t in runs_inds]

trajectory_format = ".xtc" # ".trr" or ".xtc"
topology_format = ".gro" # ".tpr" or ".gro"
forcefield = "Madrid.ff"
rcut = None
if mdp_prefix is None:
    mdp_prefix = runs_prefix 
runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]    
runs_gro = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix_gro]    
if rcut: # this skipped only if rcut is None
    rcut = 10*rcut
#---------------------------------------------------
# get_dt() takes the .mdp file
dt = get_dt(f"{path_Gromacs}{mdp_prefix}.mdp", trajectory_format)
Charges = get_Charges([cation_itp, anion_itp, solvent_itp], path_Gromacs, forcefield=forcefield)


idx = 0
run = runs[idx]
run_gro = runs_gro[idx]
u = mda.Universe(f"{path_Gromacs}{run_gro}{topology_format}",
                    f"{path_Gromacs}{run_gro}{trajectory_format}")    
print(u)                         
box=u.dimensions
center = box[0:3]/2

Nsteps = len(u.trajectory)
print(f"Number of steps: {Nsteps:.0f}")                        
# tiempo en ps    
t = np.arange(Nsteps)*dt
# Arreglo EFG:
# EFG.shape --> (NtimeSteps, Ncations, 6). The "6" is for the non-equiv. EFG matrix elements
EFG_anion = np.zeros([Nsteps, Ncations, 6])    
EFG_cation = np.zeros([Nsteps, Ncations, 6])
EFG_solvent = np.zeros([Nsteps, Ncations, 6])    

#===================================== para el GIF
# Crear listas vacías para almacenar x y y en cada iteración
# Crear listas para almacenar x, y para cada tipo de átomo en cada frame
x_gif = {atom: [] for atom in Charges['AtomType']}
y_gif = {atom: [] for atom in Charges['AtomType']}
#=====================================
t_ind=-1

for timestep in u.trajectory[:10000:10]:
    t_ind+=1
    n_frame = u.trajectory.frame
    t_print=(Nsteps//20)
    if t_ind%t_print==0:                
        print("++++++++++++++++++++++++++++++++++++++++++")
        print(f"dataset {idx+1}/{len(runs)}, frame={n_frame}, time = {t[t_ind]:.2f} ps ({t_ind/Nsteps*100:.1f} %)\n\n")                

    cations_group = u.select_atoms(f"name {cation}*")


    cation_index = -1 # indice de atomo de litio    
    for cation_in_group in cations_group:
        cation_index += 1            
        # aqui guarfo el EFG sobre el n-esimo Li al tiempo t:
        EFG_t_nthcation_anion = np.zeros([6])
        EFG_t_nthcation_cation = np.zeros([6])
        EFG_t_nthcation_solvent = np.zeros([6])
        for residue, AtomType in zip(Charges['residue'],
                                    Charges['AtomType']):
            q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]

            if np.abs(q)<1e-5: # si una especie no tiene carga, no calculo nada
                continue
            
            group = u.select_atoms(f"name {AtomType}")
            #print(residue, AtomType, q, group.n_atoms)            
            if group.n_atoms==0: continue
            # Calculate distances------------------
            # To put the cation in the center of the universe, we need this shift:
            shift = cation_in_group.position - center
            # The position of the cation is, of course, the center:
            cation_in_center = cation_in_group.position - shift        
            # Redefine coordinates with respect to the cation:
            group_newpositions = group.positions - shift
            # apply PBC to keep the atoms within a unit-cell-length to lithiu
            group_newpositions_pbc = mda.lib.distances.apply_PBC(group_newpositions,
                                                                box=box,
                                                                backend='openMP')
            # The distances to the nth-cation is the distance from the center:
            r_distances = mda.lib.distances.distance_array(cation_in_center,
                                                        group_newpositions_pbc,
                                                        backend='openMP')[0,:]
            x_gif[AtomType].append(r_distances)
            y_gif[AtomType].append(np.abs(q)/(r_distances)**3)            

            # x_distances, y_distances, z_distances = (group_newpositions_pbc-cation_in_center).T

            # if rcut: # this is skipped only if rcut is None
            #     # paso rcut de nm a Ang                                                
            #     x_distances = x_distances[r_distances<rcut]
            #     y_distances = y_distances[r_distances<rcut]
            #     z_distances = z_distances[r_distances<rcut]
            #     r_distances = r_distances[r_distances<rcut]                    
            
            # if cation in AtomType:    
            #     # Quito la distancia cero, i.e, entre la "autodistancia"
            #     # correccion:
            #     #    Cambio:
            #     #       x_distances = x_distances[r_distances!=0]
            #     #    por:
            #     #       x_distances = x_distances[r_distance > 1e-5]
            #     # De esta manera, me quito de encima numeros que son
            #     # distintos de 0 por error numerico
            #     x_distances = x_distances[r_distances>1e-5]
            #     y_distances = y_distances[r_distances>1e-5]
            #     z_distances = z_distances[r_distances>1e-5]
            #     r_distances = r_distances[r_distances>1e-5]
        
            # if (r_distances<1).any():
            #     msg = f"hay un {AtomType} a menos de 1 A del {cation}_{cation_index}"
            #     raise Warning(msg)
            # # Calculate EFG
            # EFG_t_AtomType = calculate_EFG(q,
            #                             r_distances,
            #                             x_distances,
            #                             y_distances,
            #                             z_distances)




#%% Configuración inicial de la figura
fig, ax = plt.subplots()
ax.set_xlim(1, 17)   # Ajusta el rango de x según tus necesidades
ax.set_ylim(1e-4, 0.2)   # Ajusta el rango de y según tus necesidades
# ax.set_yscale('log')

colors = ['k', 'b', 'r']
atom_types = ['MW', 'HW1', 'HW2']
# Crear una línea para cada tipo de átomo y asignarle color y leyenda
lines = []
for i, atom in enumerate(atom_types):
    line, = ax.plot([], [], 'o', color=colors[i], label=atom)  # Crear línea con marcador 'o'
    lines.append(line)

ax.legend()  # Añadir la leyenda

# Función de inicialización para el gráfico
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Función de actualización para cada frame usando los datos guardados
def update(frame):
    for i, atom in enumerate(atom_types):
        x = x_gif[atom][frame]
        y = y_gif[atom][frame]
        lines[i].set_data(x, y)  # Actualizar datos de cada tipo de átomo
    return lines

# Crear la animación
ani = FuncAnimation(fig, update, frames=len(x_gif[Charges['AtomType'][0]]), init_func=init, blit=True)

# Guardar el GIF
ani.save("animacion.gif", writer=PillowWriter(fps=10))
# %%
