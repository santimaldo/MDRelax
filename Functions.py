import numpy as np
# import matplotlib.pyplot as plt
# import MDAnalysis as mda
import pandas as pd
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.signal import correlate 
import time


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
def calculate_EFG(q, r, x, y, z):
    """
    Calculate de EFG based on charge points.
    r is the distance between observation point and EFG source (q).
    x, y, z are the components of vec(r).
    """
    dist3inv = 1/(r*r*r)
    dist5inv = 1/((r*r*r*r*r))
                    
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
    
    EFG = np.array([[Vxx, Vxy, Vxz],
                    [Vxy, Vyy, Vyz],
                    [Vxz, Vyz, Vzz]]) 

    return EFG
#=========================================================
#=========================================================




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
