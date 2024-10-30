import matplotlib.pyplot as plt
import numpy as np


path = "/home/santi/MD/MDRelax_results/Li-water/long/"

for t in range(6,7):
    file = f"EFG_total_HQ.{t*1000:.0f}_ps.long"
    solvent = "Water"
    salt = r"$Li^+$"
    
    data = np.loadtxt(path+file+".dat")
    # data = data[:,:]
    tau = data[:,0]     
    efgs = data[:,1:]
    # FIGURA: Autocorrelaciones        
    fig, ax = plt.subplots(num=t)
    label = ["Vxx","Vyy","Vzz","Vxy","Vxz","Vyz"]
    for ii in range(6):
        ax.plot(tau, efgs[:,ii], 'o-', label=label[ii], lw=3)#, color='k')

    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"EFG $[e\AA^{-3} / 4\pi\varepsilon_0]$", fontsize=16)
    ax.set_xlabel(r"$t$ [ps]", fontsize=16)    
    ax.set_xlim(-0.01, 1)
    # ax.set_ylim(0.006, 0.025)
    ax.legend()
    fig.suptitle(fr"{solvent}-{salt} EFG", fontsize=16)
    fig.tight_layout()


    Vsquarred = np.mean(np.sum(efgs*efgs, axis=1))
    print(f"<V^2>: {Vsquarred:.3e}")
    Vsquarred = np.sum(np.mean(efgs*efgs, axis=0), axis=1)
    print(f"<V^2>: {Vsquarred:.3e}")
# fig.savefig(f"{path}/Figuras/ACF_mean-over-runs_zoom.png")



# %%
