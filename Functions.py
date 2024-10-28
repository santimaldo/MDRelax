import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.signal import correlate 
from pathlib import Path
import time


#=========================================================
#=========================================================
def get_Charges(species_list, path, forcefield="park.ff"):
    """    
    get charges of the species
    Parameters
    ----------
    species_list : list of strings
        The name of the *.itp files where the atoms' parameters
        are stored. 
        E.g: ["Li", "S6", "DME_7CB8A2"]

    Returns
    -------
    Charges : dataFrame ['residue', 'AtomType',  'Charge']
    """    

    charges_df = pd.DataFrame(columns=["residue", "AtomType", "Charge"])#, "atom", "residue"])

    if not(type(species_list) is list):
        species_list = [species_list]        
    for species in species_list:
        # si no hay anion, salteo ese tipo de atomo
        if "none" in species.lower(): continue
        with open(f"{path}{forcefield}/{species}.itp", "r") as f:    
            with open(f"{path}/{species}.charges", "w") as wf:
                for ii in range(1000):        
                    condition_i = "[ atoms ]" in f.readline()
                    if condition_i: 
                        # guardo los headers
                        wf.write(f.readline()[1:]) #[1:] es para que no escriba el ";"
                        break        
                for ii in range(50):
                    line = f.readline()
                    try:            
                        condition = line.split()[0].isnumeric()            
                    except: # si la fila esta en blanco, corta
                        condition = False
                    if condition:
                        wf.write(line)
                    else:
                        break
        # esto se basa en que el atomo y la carga son las columnas 4 y 6
        charges_df_species = pd.read_csv(f"{path}/{species}.charges", sep='\s+', header=None, skiprows=1)
        charges_df_species = charges_df_species.iloc[:, [3,4,6]].drop_duplicates()
        charges_df_species.columns =["residue", "AtomType", "Charge"]
        charges_df = pd.concat([charges_df, charges_df_species], ignore_index=True, axis=0)
    return charges_df    
#=========================================================
#=========================================================


#=========================================================
#=========================================================
def get_dt(mdp_file, trajectory_format):
    """    
    get time-step between frames from the .mdp file
    Parameters
    ----------
    mdp_file : str   e.g:  '/home/MD/HQ.mdp'

    Returns
    -------
    dt : float   -  dt in ps = 1e-12 s
    """            
    if "trr" in trajectory_format:
        keyword = "nstxout"
    elif "xtc" in trajectory_format:
        keyword = "nstxout-compressed"
    with open(mdp_file, "r") as f:
        parameters_obtained=0
        for ii in range(1000):
            line = f.readline()        
            if "dt" in line:
                try:
                    # paso de dinamica
                    dt_MD = float(line.split()[2])                    
                    parameters_obtained+=1
                except:
                    msg = f"Could not get dt from 'dt' in {mdp_file}"
                    raise Warnint(msg)                
            if line.startswith(keyword):
                try:
                    # intervalo de guardado:
                    dS = float(line.split()[2])
                    parameters_obtained+=1
                except:
                    msg = f"Could not get dt from 'nstxout' in {mdp_file}"
                    raise Warnint(msg)                    
            if parameters_obtained == 2:
                break            
        if ii>=999:
            msg = f"Is this even a .mdp file? : {mdp_file}"
            raise Warning(msg)
    dt = dS*dt_MD
    print(f"MD sampling frequency: {dt} ps")
    return dt
#=========================================================
#=========================================================


#=========================================================
#=========================================================
def get_Ntimes(EFG_file):
    """    
    get number of time-steps of an EFG file
    Parameters
    ----------
    EFG_file : str   e.g:  '/home/MD/MDRelax_results/EFG_Li_6000_ps.dat'

    Returns
    -------
    Ntimes : int   -  Number of EFG time-steps
    """            
    data = np.loadtxt(EFG_file)
    Ntimes = data.shape[0]
    return Ntimes
#=========================================================
#=========================================================


#=========================================================
#=========================================================
def cumulative_simpson(ydata, x=None, initial=0):
    """
    Compute cumulative integra with simpson's rule
    ydata must be a 1D array
    """
    # inicializo    
    if x is None: x=np.arange(ydata.size)        
    integral = np.zeros_like(ydata)
    integral[0] = initial
    for nf in range(1, ydata.size):         
        ytmp, xtmp = ydata[:nf+1], x[:nf+1]    
        integral[nf] = simpson(ytmp, x=xtmp)    
    return integral
#=========================================================
#=========================================================


#=========================================================
#=========================================================
def calculate_EFG(q, r, x, y, z):
    """
    Calculate de EFG based on charge points.
    r is the distance between observation point and EFG source (q).
    x, y, z are the components of vec(r).
    """
    dist3inv = 1/(r*r*r)
    dist5inv = 1/(r*r*r*r*r)
                    
    Vxx = q *(3 * x * x * dist5inv - dist3inv)
    Vxy = q *(3 * x * y * dist5inv )
    Vxz = q *(3 * x * z * dist5inv )
    Vyy = q *(3 * y * y * dist5inv - dist3inv)
    Vyz = q *(3 * y * z * dist5inv )
    Vzz = q *(3 * z * z * dist5inv - dist3inv)

    Vxx = np.sum(Vxx)
    Vxy = np.sum(Vxy)
    Vxz = np.sum(Vxz)
    Vyy = np.sum(Vyy)
    Vyz = np.sum(Vyz)
    Vzz = np.sum(Vzz)
    
    # This was HIGHLY memory consuming
    # EFG = np.array([[Vxx, Vxy, Vxz],
    #                 [Vxy, Vyy, Vyz],
    #                 [Vxz, Vyz, Vzz]]) 
    # Replaced by:
    EFG = np.array([Vxx, Vyy, Vzz, Vxy, Vxz, Vyz])

    return EFG
#=========================================================
#=========================================================

#=========================================================
#=========================================================
def get_EFG_data(path_Gromacs, path_MDrelax,
                 species = ["cation", "anion", "solvent"],
                 species_itp = ["Li","none", "DME_7CB8A2"],
                 Ncations = 1,
                 mdp_prefix =None,
                 runs_prefix = "HQ",
                 runs_suffix = None,
                 runs_suffix_gro = None,
                 trajectory_format = ".trr",
                 topology_format = ".gro",
                 forcefield="park.ff"
                 ):
    """
    Function for reading GROMACS data and calculating the EFG
    at the cations' positions from different EFG-sources (other charges)
    """
    # Unpacking variables:
    cation, anion, solvent = species # names
    cation_itp, anion_itp, solvent_itp = species_itp # as in .itp files
    if mdp_prefix is None:
        mdp_prefix = runs_prefix 
    runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]    
    runs_gro = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix_gro]    
    #---------------------------------------------------
    # get_dt() takes the .mdp file
    dt = get_dt(f"{path_Gromacs}{mdp_prefix}.mdp", trajectory_format)
    Charges = get_Charges([cation_itp, anion_itp, solvent_itp], path_Gromacs, forcefield=forcefield)

    # loop over different runs
    for idx in range(len(runs)):    
        run = runs[idx]
        run_gro = runs_gro[idx]
        u = mda.Universe(f"{path_Gromacs}{run_gro}{topology_format}",
                         f"{path_Gromacs}{run_gro}{trajectory_format}")    
        print(u)                         
        box=u.dimensions
        center = box[0:3]/2
                
        # tiempo en ps    
        t = np.arange(len(u.trajectory))*dt
        # Arreglo EFG:
        # EFG.shape --> (NtimeSteps, Ncations, 3, 3)        
        EFG_anion = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        EFG_cation = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        EFG_solvent = np.zeros([len(u.trajectory), Ncations, 3, 3])    
        t_ind = -1
        for timestep in u.trajectory:
            t_ind+=1         
            n_frame = u.trajectory.frame
            if t_ind%100==0:                
                print("++++++++++++++++++++++++++++++++++++++++++")
                print(f"dataset {idx+1}/{len(runs)}, frame={n_frame}, time = {t[t_ind]:.2f} ps\n\n")                

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

                    if q<1e-5: # si una especie no tiene carga, no calculo nada
                        continue
                    
                    group = u.select_atoms(f"name {AtomType}")                               
                    if group.n_atoms==0: continue
                    # Calculate distances------------------
                    # Put Li in the center of the universe
                    cation_in_center = cation_in_group.position - center
                    # Redefine coordinates with respect to lithium:
                    group_newpositions = group.positions - cation_in_center
                    # apply PBC to keep the atoms within a unit-cell-length to lithiu
                    group_newpositions_pbc = mda.lib.distances.apply_PBC(group_newpositions,
                                                                        box=box,
                                                                        backend='openMP')
                    # The distances to the nth-cation is the distance from the center:
                    r_distances = mda.lib.distances.distance_array(center,
                                                                group_newpositions_pbc,
                                                                backend='openMP')[0,:]
                    x_distances, y_distances, z_distances = (group_newpositions_pbc-center).T
                    
                    if cation in AtomType:    
                        # Quito la distancia cero, i.e, entre la "autodistancia"
                        # correccion:
                        #    Cambio:
                        #       x_distances = x_distances[r_distances!=0]
                        #    por:
                        #       x_distances = x_distances[r_distance > 1e-5]
                        # De esta manera, me quito de encima numeros que son
                        # distintos de 0 por error numerico
                        x_distances = x_distances[r_distances>1e-5]
                        y_distances = y_distances[r_distances>1e-5]
                        z_distances = z_distances[r_distances>1e-5]
                        r_distances = r_distances[r_distances>1e-5]
                
                    if (r_distances<1).any():
                        msg = f"hay un {AtomType} a menos de 1 A del {cation}_{cation_index}"
                        raise Warning(msg)
                    # Calculate EFG
                    EFG_t_AtomType = calculate_EFG(q, r_distances, x_distances,
                                            y_distances, z_distances)
                        
                    if anion in residue:
                        EFG_t_nthcation_anion += EFG_t_AtomType                    
                    elif cation in residue:
                        EFG_t_nthcation_cation += EFG_t_AtomType
                    else:
                        EFG_t_nthcation_solvent += EFG_t_AtomType                    
                EFG_anion[t_ind, cation_index, :, :] = EFG_t_nthcation_anion
                EFG_cation[t_ind, cation_index, :, :] = EFG_t_nthcation_cation
                EFG_solvent[t_ind, cation_index, :, :] = EFG_t_nthcation_solvent
        #---------------------------------------------------------
        ### EXPORT DATA        
        EFG_total = EFG_cation + EFG_anion + EFG_solvent
        EFGs = [EFG_cation, EFG_anion, EFG_solvent, EFG_total]
        efg_sources = [cation, anion, solvent, "total"]
        for EFG, efg_source in zip(EFGs, efg_sources):
            Vxx, Vyy, Vzz = EFG[:,:,0,0], EFG[:,:,1,1], EFG[:,:,2,2]
            Vxy, Vyz, Vxz = EFG[:,:,0,1], EFG[:,:,1,2], EFG[:,:,0,2]
            
            data = [t]
            for nn in range(cations_group.n_atoms):    
                data += [Vxx[:,nn], Vyy[:,nn], Vzz[:,nn], Vxy[:,nn], Vyz[:,nn], Vxz[:,nn]]
            
            data = np.array(data).T    
            header =  r"# t [fs]\t Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t"\
                    r"Li2:  Vxx, Vyy, Vzz, Vxy, Vyz, Vxz \t and so on..."            
            filename = f"{path_MDrelax}/EFG_{efg_source}_{run}.dat"
            np.savetxt(filename, data, header=header)
        del EFG_anion, EFG_solvent
        #----------------------------------------------------------    
    return 0
#=========================================================
#=========================================================

#=========================================================
#=========================================================
def Autocorrelate(tau, EFG, method='scipy'):
    """
    parameters:
    tau :   1d-array    -  times in picoseconds
    EFG:    array       -  shape: (3,3, Ntimes, Nruns, Ncations)
    """
    dt = tau[1]-tau[0]
    _, _, Ntau, Nruns, Ncations = EFG.shape    
    # initialize
    ACF = np.zeros([Ntau, Nruns, Ncations])
    # efg_squared = np.zeros([Ntau, Nruns, Ncations])
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Metodo scipy.signal.correlate
    if 'scipy' in method.lower():
        print("ACF method: scipy")
        # loop over EFG matrix elements:        
        for kk in range(Nruns):
            for ll in range(Ncations):  
                acf = np.zeros(Ntau)                      
                for ii in range(3):
                    for jj in range(3):
                        if ii<jj:
                            factor=2
                        elif ii==jj:
                            factor=1
                        elif jj<ii:
                            continue                            
                        V = EFG[ii,jj,:,kk,ll]
                        corr = factor*correlate(V,V, mode='full')
                        # take only positive time lags:
                        acf += corr[corr.size//2:]
    #### MAL!!!! ESTOY HACIENDO LA SUMA DE CORRELACIONES!!!
    # TENGO QUE HACER LA CORRELACION DE LA SUMA!                        
                ACF[:,kk,ll] = acf
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Metodo manual que yo implemente
    else:
        # loop over time:        
        for jj in range(Ntau):        
            if jj%1000==0:
                ttn = time.time()
                print(f"    tau: {tau[jj]} ps")                        
            max_tau_index = tau.size-jj                        
            acf_jj = np.zeros([Nruns, Ncations]) 
            for ii in range(0,max_tau_index):                                 
                acf_jj += np.sum(EFG[:,:,ii,:,:]*EFG[:,:,ii+jj,:,:],
                                axis=(0,1))                    
            ACF[jj,:,:] = acf_jj/(max_tau_index+1)           
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = =             
    return ACF
#=========================================================
#=========================================================   


#=========================================================
#=========================================================
def calculate_ACF(path_MDrelax,
                  savepath = None,
                  species = ["cation", "anion", "solvent"],                  
                  Ncations = 1,
                  runs_prefix = "HQ",
                  runs_suffix = None,
                  method='scipy'                                    
                  ):
    """
    Reads EFG files and calculate ACF functions, plot the results
    and export plots
    """
    print("="*33)
    print("="*10+f"calculate_ACF"+"="*10)
    print("="*33)
    if savepath is None:
        savepath = path_MDrelax 
    # unpack species:
    cation, anion, solvent = species
    # creates the savepath if it doesn't exist
    Path(f"{savepath}/Figures").mkdir(parents=True, exist_ok=True)
    runs = [f"{runs_prefix}{runsuf}" for runsuf in runs_suffix]
    # Number of time steps
    Ntimes = get_Ntimes(f"{path_MDrelax}EFG_{cation}_{runs[0]}.dat")    
    # Number of runs
    Nruns = len(runs)
    
    
    t0 = time.time()
    efg_sources = [cation, anion, solvent, "total"]
    EFG_cation = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_anion = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_solvent = np.zeros([3,3, Ntimes, Nruns, Ncations])
    EFG_total = np.zeros([3,3, Ntimes, Nruns, Ncations])
    acf_cation = np.zeros([Ntimes, Nruns, Ncations])
    acf_anion = np.zeros([Ntimes, Nruns, Ncations])
    acf_solvent = np.zeros([Ntimes, Nruns, Ncations])
    acf_total = np.zeros([Ntimes, Nruns, Ncations])
    efg_variance = np.zeros([Nruns, Ncations])
    acf_total_mean = np.zeros([Ntimes, Nruns])
    run_ind = -1
    print("Reading files...")
    # Loop sobre runs para calcular ACF
    for run in runs:
        print(f"RUN: {run} "+"="*30)
        run_ind += 1    
        efg_source_index = -1
        for efg_source in efg_sources: 
            efg_source_index += 1                
            filename = f"{path_MDrelax}/EFG_{efg_source}_{run}.dat"
            print("reading ", filename)
            data = np.loadtxt(filename)[:Ntimes, :]        
            # read the time only once:
            if (run_ind==0) and (efg_source_index==0):
                # tau and t are the same
                tau = data[:,0]            
            # data columns order:    
            # t; Li1: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz; Li2: Vxx, Vyy, Vzz, Vxy, Vyz, Vxz..
            # 0; Li1:   1,   2,   3,   4,   5,   6; Li2:   7,   8,   9,  10,  11,  12..        
            t0 = time.time()
            for nn in range(Ncations): #uno para cada litio
                # plt.plot(data[:,0], data[:,nn+1])            
                Vxx, Vyy, Vzz = data[:,1+nn*6], data[:,2+nn*6], data[:,3+nn*6]
                Vxy, Vyz, Vxz = data[:,4+nn*6], data[:,5+nn*6], data[:,6+nn*6]                                            
                # this EFG_nn is (3,3,Ntimes) shaped
                # nn is the cation index
                EFG_nn = np.array([[Vxx, Vxy, Vxz],
                                    [Vxy, Vyy, Vyz],
                                    [Vxz, Vyz, Vzz]])                                
                # EFG_{source} shape: (3,3, Ntimes, Nruns, Ncations)
                if cation in efg_source:            
                    EFG_cation[:,:,:, run_ind, nn] = EFG_nn            
                elif anion in efg_source:            
                    EFG_anion[:,:,:, run_ind, nn] = EFG_nn            
                elif solvent in efg_source:
                    EFG_solvent[:,:,:, run_ind, nn] = EFG_nn
                elif "total" in efg_source:
                    EFG_total[:,:,:, run_ind, nn] = EFG_nn 

    #Calculo ACF========================================================
    print("Calculating ACF...")        
    #-------------------------------------------
    # calculate total ACF:
    print(rf"ACF with EFG_total")        
    acf_total = Autocorrelate(tau, EFG_total, method=method)
    #-------------------------------------------
    # calculate only if a cation is a possible efg source
    print(rf"ACF with EFG-source: {cation}")
    if Ncations>1:     
        acf_cation = Autocorrelate(tau, EFG_cation, method=method)
    else:
        print(rf"There is only one {cation}. It can't be an EFG source")
    #-------------------------------------------
    # calculate only if anions exist
    print(rf"ACF with EFG-source: {anion}")
    if "none" in anion.lower():
        print("since no anion is present, this step is skipped")
    else:    
        acf_anion = Autocorrelate(tau, EFG_anion, method=method)
    #-------------------------------------------
    print(rf"ACF with EFG-source: {solvent}")
    acf_solvent = Autocorrelate(tau, EFG_solvent, method=method)
    tn = time.time()
    print(f"tiempo=   {tn-t0} s") 
    #===================================================================    
    # EFG variances
    #-------------------------------------------
    # calculating variance:
    efg_squared = np.sum(EFG_total*EFG_total, axis=(0,1))
    # efg_variance is the squared efg averaged over time:
    # shape: (Nruns, Ncations) mean over time
    efg_variance = np.mean(efg_squared, axis=0)
    #-------------------------------------------        
    acf_total_mean = np.mean(acf_total, axis=2)
    acf_cation_mean = np.mean(acf_cation, axis=2)
    acf_anion_mean = np.mean(acf_anion, axis=2)
    acf_solvent_mean = np.mean(acf_solvent, axis=2)
    
    # calculating cross product terms of ACF
    acf_cross = acf_total - (acf_cation+acf_anion+acf_solvent)
    acf_cross_mean = np.mean(acf_cross, axis=2)

    # Saving Data =======================================
    # 0) EFG variance:  mean over runs and cations
    # 1) ACF:           mean over runs and cations
    # 2) EFG variance:  mean over cations, one per run
    # 3) ACF:           mean over cations, one per run
    # 4) ACF:           one per run, all cations data

    # 0)-------------------------------------------------
    ## save efg_variance_mean_over_runs data
    efg_variance_mean_over_runs = np.mean(efg_variance, axis=(0,1))
    header = f"EFG variance: mean over runs.\t"\
              "Units: e^2*A^-6*(4pi*epsilon0)^-2"    
    np.savetxt(f"{savepath}/EFG_variance_mean-over-runs.dat", [efg_variance_mean_over_runs], header=header)    
    # 1)-------------------------------------------------
    ## save acf_mean_over_runs data
    data = np.array([tau, 
                     np.mean(acf_total_mean, axis=1),
                     np.mean(acf_cation_mean, axis=1),
                     np.mean(acf_anion_mean, axis=1),
                     np.mean(acf_solvent_mean, axis=1),
                     np.mean(acf_cross_mean, axis=1)]).T
    header = f"tau\tACF_total\tACF_{cation}\tACF_{anion}"\
             f"\tACF_{solvent}\tACF_cross-terms\n"\
              "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
    np.savetxt(f"{savepath}/ACF_mean-over-runs.dat", data, header=header)
    #------------------------------------------------------
    run_ind = -1
    for run in runs:    
        run_ind += 1  
        # 2)-----------------------------------------------
        # guardo varianzas promedio    
        data = efg_variance[run_ind,:]
        header = f"EFG variance for each {Ncations} {cation} cations.\t"\
                "Units: e^2*A^-6*(4pi*epsilon0)^-2"
        np.savetxt(f"{savepath}/EFG_variance_{run}.dat", data, header=header)
        # 3)-----------------------------------------------
        # guardo autocorrelaciones promedio
        data = np.array([tau, 
                        acf_total_mean[:, run_ind],
                        acf_cation_mean[:, run_ind],
                        acf_anion_mean[:, run_ind], 
                        acf_solvent_mean[:, run_ind],
                        acf_cross_mean[:, run_ind]]).T
        header = f"tau\tACF_total\tACF_{cation}\tACF_{anion}"\
                 f"\tACF_{solvent}\tACF_cross-terms\n"\
                  "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
        np.savetxt(f"{savepath}/ACF_{run}.dat", data, header=header)
        # 4)-----------------------------------------------
        for acf, source in zip([acf_total, acf_cation,
                            acf_anion, acf_solvent],
                           ["total", cation, anion, solvent]):                   
            data = [acf[:, run_ind, nn] for nn in range(Ncations)] 
            data = np.array([tau]+ data).T
            header = f"tau\tACF_total for each cation\n"\
                    "Units: time=ps,   ACF=e^2*A^-6*(4pi*epsilon0)^-2"
            np.savetxt(f"{savepath}/ACF_efgsource-{source}_{run}_all-cation.dat", data, header=header)
    return 0
#=========================================================
#=========================================================


# OLD
# #=========================================================
# #=========================================================
# def Autocorrelate(tau, EFG, method='scipy'):
#     """
#     parameters:
#     tau :   1d-array    -  times in picoseconds
#     EFG:    array       -  shape: (3,3, Ntimes, Nruns, Ncations)
#     """
#     dt = tau[1]-tau[0]
#     _, _, Ntau, Nruns, Ncations = EFG.shape    
#     # initialize
#     acf = np.zeros([Ntau, Nruns, Ncations])
#     efg_squared = np.zeros([Ntau, Nruns, Ncations])
#     # loop over time:        
#     for jj in range(Ntau):        
#         if jj%1000==0:
#             ttn = time.time()
#             print(f"    tau: {tau[jj]} ps")            
#         tau_jj = jj*dt          
#         max_tau_index = tau.size-jj                        
#         acf_jj = np.zeros([Nruns, Ncations]) 
#         for ii in range(0,max_tau_index):                                 
#             acf_jj += np.sum(EFG[:,:,ii,:,:]*EFG[:,:,ii+jj,:,:],
#                              axis=(0,1))                    
#         acf[jj,:,:] = acf_jj/(max_tau_index+1)
#         tt0 = time.time()
#     return acf
# #=========================================================
# #=========================================================    
