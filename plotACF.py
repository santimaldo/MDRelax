import matplotlib.pyplot as plt
import numpy as np




path = "/home/santi/MD/MDRelax_results/Li-water/long/"

for t in range(6,8):
    file = f"ACF_HQ.{t*1000:.0f}_ps.long"
    solvent = "Water"
    salt = r"$Li^+$"

    data = np.loadtxt(path+file+".dat")
    tau = data[:5000,0]
    ACF = data[:5000,1] 
    del data
    # FIGURA: Autocorrelaciones        
    fig, ax = plt.subplots(num=t)
    ax.plot(tau, ACF, label=f"run {t}", lw=3)#, color='k')

    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"ACF $[e^2\AA^{-6}(4\pi\varepsilon_0)^{-2}]$", fontsize=16)
    ax.set_xlabel(r"$\tau$ [ps]", fontsize=16)    
    # ax.set_xlim(-0.01, 1)
    # ax.set_ylim(0.006, 0.025)
    ax.legend()
    fig.suptitle(fr"{solvent}-{salt} EFG Autocorrelation Function", fontsize=16)
    fig.tight_layout()
# fig.savefig(f"{path}/Figuras/ACF_mean-over-runs_zoom.png")

# %%
