import matplotlib.pyplot as plt
import numpy as np
from Functions import *




path = "/home/santi/MD/MDRelax_results/Li-water/long/"
        

for t in range(6,11):
    file = f"ACF_HQ.{t*1000:.0f}_ps.long"
    file_efg = f"EFG_variance_HQ.{t*1000:.0f}_ps.long"
    solvent = "Water"
    salt = r"$Li^+$"

    data = np.loadtxt(path+file_efg+".dat")
    Vsquarred = data[0]
    
    Ntaus = 10000
    Ntaus = 1000
    data = np.loadtxt(path+file+".dat")
    tau = data[:Ntaus,0]
    ACF = data[:Ntaus,1]
    del data

    print("="*50)
    print(f"run: {t}\tACF[0]: {ACF[0]:.3e}\t<V^2>: {Vsquarred:.3e}")
      

    # FIGURA: Autocorrelaciones        
    fig, ax = plt.subplots(num=t)
    ACF = ACF/max(ACF)
    ax.plot(tau, ACF, label=f"run {t}", lw=3)#, color='k')
    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
    ax.set_ylabel(r"ACF", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    ax.set_xlim(-0.01, 0.8)
    # ax.set_ylim(0.006, 0.025)
    ax.legend()
    fig.suptitle(fr"{solvent}-{salt} EFG Autocorrelation Function", fontsize=16)
    fig.tight_layout()

    



    ##
    fig, ax = plt.subplots(num=t*1000)    
    cumulative = cumulative_simpson(ACF, x=tau, initial=0)        
    ax.plot(tau, cumulative, label=f"run: {t}", lw=3)
    ax.set_ylabel(r"$C(\tau)$ [ps]", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    title = f"{solvent}-"\
            f"{salt}"+"\n"\
            r" Cumulative Integral of ACF:   "\
            r"$C(\tau)=\langle \mathrm{V}^2\rangle^{-1} \int_0^\tau $"\
            r"$\langle\sum_{\alpha\beta}V_{\alpha\beta}(0)$"\
            r"$V_{\alpha\beta}(t')\rangle dt'$"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
# fig.savefig(f"{path}/Figuras/ACF_mean-over-runs_zoom.png")

# %%
