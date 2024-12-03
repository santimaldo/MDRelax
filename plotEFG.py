import matplotlib.pyplot as plt
import numpy as np


# names.append(r"DOL-Li$^+$ 0.1 M - charmm36")
# paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/DOL_no-anion/")
# names.append(r"DME-Li$^+$ 0.1 M - charmm36")
# paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/DME_no-anion/"
# names.append(r"TEGDME-Li$^+$ 0.1 M - charmm36")
# paths.append("/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/TEGDME_no-anion/")

path = "/home/santi/MD/MDRelax_results/CHARMM/Li_no-anion/TEGDME_no-anion/"
solvent = "TEGDME"
salt = r"$Li^+$"
    
# SOLO MUESTRO EL PRIMER ION
for t in range(6,7):
    file = f"EFG_total_HQ.{t*1000:.0f}_ps"
    
    data = np.loadtxt(path+file+".dat")
    # data = data[:,:]
    tau = data[:,0]     
    efgs = data[:,1:7]
    # FIGURA: Autocorrelaciones        
    fig, ax = plt.subplots(num=t)
    label = ["Vxx","Vyy","Vzz","Vxy","Vxz","Vyz"]
    for ii in range(6):
        ax.plot(tau, efgs[:,ii], 'o-', label=label[ii], lw=3)#, color='k')

    trace = np.sum(efgs[:, :3], axis=1)
    ax.plot(tau, trace, 'k', label="Tr(V)")
    ax.axhline(0, color='k', ls='--')
    ax.set_ylabel(r"EFG $[e\AA^{-3} / 4\pi\varepsilon_0]$", fontsize=16)
    ax.set_xlabel(r"$t$ [ps]", fontsize=16)    
    # ax.set_xlim(-0.01, 1)
    # ax.set_ylim(0.006, 0.025)
    ax.legend()
    fig.suptitle(fr"{solvent}-{salt} EFG", fontsize=16)
    fig.tight_layout()


    prefactor=np.array([1,1,1,2,2,2])
    Vsquarred = np.mean(np.sum(efgs*efgs*prefactor, axis=1))
    print(f"<V^2>: {Vsquarred:.3e}")
    Vsquarred = np.sum(prefactor*np.mean(efgs*efgs, axis=0))
    print(f"<V^2>: {Vsquarred:.3e}")
# fig.savefig(f"{path}/Figuras/ACF_mean-over-runs_zoom.png")
#%%
fig, ax = plt.subplots(num=137681376813768)
mean = np.mean(trace)
std = np.std(trace)
label = "Tr(V)\n"+\
        f"mean: {mean:.2e}\n"+\
        f"std.dev.: {std:.2e}"
ax.plot(tau, trace, 'k', label=label)
ax.axhline(0, color='r', ls='--')
ax.set_ylabel(r"EFG $[e\AA^{-3} / 4\pi\varepsilon_0]$", fontsize=16)
ax.set_xlabel(r"$t$ [ps]", fontsize=16)    
ax.legend()
fig.suptitle(fr"{solvent}-{salt} EFG Trace", fontsize=16)
fig.tight_layout()



# %%
