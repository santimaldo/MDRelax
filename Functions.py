import numpy as np
# import matplotlib.pyplot as plt
# import MDAnalysis as mda
import pandas as pd


def get_Charges(species_list, path):
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
    Charges : dataFrame ['AtomType',  'Charge']
    """    

    charges_df = pd.DataFrame(columns=["AtomType", "Charge"])#, "atom", "residue"])
    for species in species_list:
        with open(f"{path}park.ff/{species}.itp", "r") as f:    
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
        charges_df_species = charges_df_species.iloc[:, 4:7:2].drop_duplicates()
        charges_df_species.columns =["AtomType", "Charge"]
        charges_df = pd.concat([charges_df, charges_df_species], ignore_index=True, axis=0)
    return charges_df    


def get_dt(mdp_file):
    """    
    get time-step between frames from the .mdp file
    Parameters
    ----------
    mdp_file : str   e.g:  '/home/MD/HQ.mdp'

    Returns
    -------
    dt : float   -  dt in ps = 1e-12 s
    """            
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
            if "nstxout" in line:
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
    return dt


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