import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson

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


path = "/home/santi/MD/GromacsFiles/2024-08_DME_3rd-test/MDRelax/"
runs_inds = range(6, 11)
MDfiles = [f"HQ.{i}" for i in runs_inds]
runs = [f"{t*1000:.0f}_ps" for t in runs_inds]

cation, anion, solvent_hr = ["Li", "S6", "DME"] # hr stands for "human readable"
savepath = path
salt = r"$Li_2S_6$"




efg_variance = np.zeros([len(runs)])
run_ind = -1
for run in runs:    
    run_ind += 1        
    data = np.loadtxt(f"{path}ACF_{run}.dat")[:,0:2]
    if run_ind == 0:
        tau = data[:,0]
        acf_means = np.zeros([tau.size, len(runs)])        
    acf_means[:, run_ind] = data[:,1]
    efg_variance[run_ind] = np.loadtxt(f"{path}EFG_variance_{run}.dat")


#%% Finally, the mean of all runs:
# FIGURA: Autocorrelaciones    
fig, ax = plt.subplots(num=3781781746813134613543546)
run_ind = -1
for run in runs:    
    run_ind += 1    
    ax.plot(tau, acf_means[:, run_ind], label=f"run: {runs[run_ind]}", 
            lw=2, color='grey', alpha=0.5)
# compute the mean over runs:            
acf_mean = np.mean(acf_means, axis=1)
ax.plot(tau, acf_mean, label=f"Mean over runs", lw=3, color='k')


ax.axhline(0, color='k', ls='--')
ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
ax.legend()
fig.suptitle(fr"{solvent_hr}-{salt} EFG Autocorrelation Function", fontsize=16)
fig.tight_layout()
fig.savefig(f"{savepath}/Figuras/ACF_mean-over-runs.png")


# FIGURA: Cumulatives---------------------------------------------------
fig, ax = plt.subplots(num=37817817174681374681354132541354)
run_ind = -1
for run in runs:    
    run_ind += 1        
    data = acf_means[:, run_ind]
    integral = cumulative_simpson(data, x=tau, initial=0)    
    cumulative = integral/efg_variance[run_ind]
    ax.plot(tau, cumulative, label=f"run: {runs[run_ind]}",
            lw=2, color="grey", alpha=0.5)                                   
    
# compute the mean over runs:            
######## ACA NO SE SI POMEDIAR LA VARIANZA ANTES O DESPUES DE INTEGRAR
data = acf_mean # mean over runs
efg_variance_mean_over_runs = np.mean(efg_variance)
integral = cumulative_simpson(data, x=tau, initial=0)
cumulative = integral/efg_variance_mean_over_runs
ax.plot(tau, cumulative, label="Mean over runs", lw=3, color="k") 

ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
title = f"{solvent_hr}-"\
        f"{salt}"+"\n"\
        r" Cumulative Integral of ACF:   "\
        r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
        r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
        r"$V_{\alpha\beta}(t')\rangle dt'$"
fig.suptitle(title, fontsize=12)
fig.tight_layout()

corr_time_range = np.array([0.4, 0.8])*cumulative.size
corr_time = np.mean(cumulative[int(corr_time_range[0]):int(corr_time_range[1])])
ax.hlines(corr_time, 
         tau[int(corr_time_range[0])],
         tau[int(corr_time_range[1])], 
         ls='--', color='grey', lw = 1.5,
         label=f"~{corr_time:.1f} ps")

ax.legend()
fig.savefig(f"{savepath}/Figuras/CorrelationTime_mean-over-runs.png")

# %%
