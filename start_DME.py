#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:33:25 2023

@author: santi

First script to test the EFG summation


TO DO LIST:

+ hacer un modulo de funciones.
+ separar el calculo de EFG por especie
+ reemplazar las listas por estructuras mas optimas como np.arrays ¿o dataframes?


"""
# ii = 0
# from MDAnalysis.analysis.rdf import InterRDF
# Li_resids = [atom.resid for atom in u.select_atoms("name Li*")]

# for Li_resid in Li_resids:
#     ii+=1
#     plt.figure(ii)
#     rdf = InterRDF(u.select_atoms(f"resid {Li_resid}"), u.select_atoms("name S6t"))
#     rdf.run()
#     plt.plot(rdf.bins, rdf.rdf)



import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from Functions import *

path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/"
species_list = ["Li", "S6", "DME_7CB8A2"]

# runs = [f"{t:.1f}_ps" for t in [6000,7000,8000,9000,10000]]
runs = [f"{t:.1f}_ps" for t in [8000]]
MDfiles = [f"HQ.{i}" for i in range(6,11)]

NPS = 2 # moleculas de polisulfuro
dt = 0.01 # ps (tiempo entre frames) ## parametro de la simulacion: automatizar.

Charges = get_Charges(species_list, path)

# loop over different runs
for idx in range(len(MDfiles)):    
    run = runs[idx]
    filename = MDfiles[idx]    
    #u = mda.Universe(f"{path}{filename}.tpr", f"{path}{filename}.trr")
    #u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.trr")
    u = mda.Universe(f"{path}{filename}.gro", f"{path}{filename}.xtc")
    box=u.dimensions
    center = box[0:3]/2
            
    # tiempo en ps
    trajectory = u.trajectory        
    t = np.arange(len(trajectory))*dt
    # Arreglo EFG:
    # EFG.shape --> (NtimeSteps, NLiAtoms, 3, 3)    
    EFG = np.zeros([len(trajectory), NPS*2, 3, 3])    
    nn = -1
    for timestep in trajectory:
        nn+=1 
        print("++++++++++++++++++++++++++++++++++++++++++")                    
        n_frame = u.trajectory.frame                
        print(f"dataset {idx+1}/{len(MDfiles)}, frame={n_frame}, time = {t[nn]:.2f} ps\n\n")                

        group_Li = u.select_atoms("name Li*")                    
        #EFG_t = pd.DataFrame[columns=Charges['AtomType']] ############### CONTINUAR ESTA IDEA
        nLi = -1 # indice de atomo de litio    
        for Li_atom in group_Li:
            nLi += 1            
            # aqui guarfo el EFG sobre el n-esimo Li al tiempo t:
            EFG_t_nLi = np.zeros([3,3])
            for AtomType in Charges['AtomType']:                                           
                q = Charges[Charges['AtomType']==AtomType]['Charge'].values[0]
                
                group = u.select_atoms(f"name {AtomType}")                                
                # Calculate distances------------------
                # Put Li in the center of the universe
                Li_in_center = Li_atom.position - center
                # Redefine coordinates with respect to lithium:
                group_newpositions = group.positions - Li_in_center            
                # apply PBC to keep the atoms within a unit-cell-length to lithiu
                group_newpositions = mda.lib.distances.apply_PBC(group_newpositions,
                                                                    box=box,
                                                                    backend='openMP')                                                            
                if 'li' not in AtomType.lower():
                    r_distances = mda.lib.distances.distance_array(center, 
                                                               group_newpositions)        
                    x_distances, y_distances, z_distances = (group_newpositions-center).T                      
                else:
                    # Defino la distancia de otra forma, para poder quitar
                    # la "autodistancia" del litio observado                    
                    x_distances, y_distances, z_distances = (group_newpositions-center).T                     
                    r_distances = np.sqrt(x_distances*x_distances+
                                          y_distances*y_distances+
                                          z_distances*z_distances)
                    # Quito la distancia cero, i.e, entre la "autodistancia"
                    x_distances = x_distances[r_distances!=0]
                    y_distances = y_distances[r_distances!=0]
                    z_distances = z_distances[r_distances!=0]
                    r_distances = r_distances[r_distances!=0]
        
                # Calculate EFG--------------------------------------------------------
                EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                        y_distances, z_distances)
                      
                # EFG_t = [EFG_t[kk]+EFG_t_AtomType[kk] for kk in range(6)]
                EFG_t_nLi += EFG_t_AtomType                            
            EFG[nn, nLi, :, :] = EFG_t_nLi                    
    #---------------------------------------------------------
    ### EXPORT DATA    
    Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
    Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
    
    data = [t]
    for nn in range(group_Li.n_atoms):    
        data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
    
    data = np.array(data).T    
    header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
              r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
    filename = f"{path}/MDRelax/EFG_{run}.dat"
    np.savetxt(filename, data, header=header)
    del EFG
    #----------------------------------------------------------    
    
    #---------------
    #Calculo el profucto de EFG a tiempo t y a tiempo 0,
    ### esta vez variando cual es el tiempo 0 (promedio en ensamble)
    
    ### COMENTO ESTO PARA HACER UNA CORRIDA LARGA:

    # dtau = np.diff(t)[0]
    # efg = EFG
    # acf = np.zeros([t.size, group_Li.n_atoms])
    # Num_promedios = np.zeros(t.size)
    # for ii in range(t.size):    
    #     tau = ii*dtau
    #     jj, t0, acf_ii = 0, 0, 0
    #     while t0+tau<=t[-1]:
    #         print(f"tau = {tau} ps, t0 = {t0:.2f} ps, ---------{jj}")                
    #         acf_ii += np.sum(efg[ii,:,:,:]*efg[ii+jj,:,:,:], axis=(1,2))
    #         jj+=1
    #         t0 = jj*dtau
    #     print(f"el promedio es dividir por {jj}")
    #     acf[ii,:] = acf_ii/jj
    #     Num_promedios[ii] = jj
    # #
    # plt.figure(44847)
    # plt.plot(t, Num_promedios,'k--')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    
    # plt.ylabel(r"Numero de promedios",
    #            fontdict={'fontsize':16})
    
    # plt.figure(44848)
    # for jj in range(group_Li.n_atoms):        
    #     # plt.plot(t, ACF[:,jj]/ACF[0,jj], 'o--', 
    #              # label=f'Li {jj+1}, Sin promediar')   
    #     plt.plot(t, acf[:,jj], 'o-', 
    #              label=rf'$Li_{jj+1}$')
            
    # plt.plot(t, np.mean(acf, axis=1),'o-', 
    #          label = r'mean')
    
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    
    # plt.ylabel(r"$\langle\ EFG(t)\cdot EFG(0)\ \rangle_t$",
    #            fontdict={'fontsize':16})
    
    # plt.xlim([0,t[-1]*0.5])
    # plt.gca().axhline(0, color='k', ls='--')
    # # plt.gca().axhline(1, color='k', ls='--')
    # plt.tight_layout()
    # plt.legend(loc='center right')
    # plt.savefig(f"{path}/MDRelax/ACF_{run}.png")
    # plt.show()
    
    # #
    # plt.figure(44849)
    # plt.plot(t, np.mean(acf, axis=1)/np.mean(acf, axis=1)[0],'o-', color='green', 
    #          label = r'Promedio entre atomos de Li y en $<>_t$')
    # plt.xlabel("Time [ps]", fontdict={'fontsize':16})
    # plt.ylabel("Autocorrelation Function", fontdict={'fontsize':16})
    # plt.gca().axhline(0, color='k', ls='--')
    # plt.gca().axhline(1, color='k', ls='--')
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f"{path}/MDRelax/ACFnorm_{run}.png")
    # plt.show()    