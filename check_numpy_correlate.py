import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

t = np.linspace(0,10,512)
V = np.exp(-t)


t0 = time.time()
# Calculo ACF manual
Ntau = t.size                               
tau = np.zeros([Ntau])
acf = np.zeros([Ntau])        
dt = t[1]-t[0]
for jj in range(Ntau):    
    tau_jj = jj*dt            
    max_tau_index = t.size-jj            
    # jj, t0, acf_ii = 0, 0, 0            
    # while t0+tau<=times[-1]:
    acf_jj = 0         
    for ii in range(0,max_tau_index):
        # if ii%100==0:
        #     print(f"tau = {tau_jj:.2f} fs, t0 = {ii*dt:.2f} ps,"\
        #           f" EFG[{ii}]*EFG[{ii+jj}]")                                
        acf_jj +=  V[ii]*V[ii+jj]                                        
    tau[jj] = tau_jj
    promedio = acf_jj/(max_tau_index+1)
    acf[jj] = promedio
    
tf = time.time()
print(f"calculo manual demora: {tf-t0} s")


# def acorr(x):

#     result = numpy.correlate(x, x, mode='full')
#     return result[result.size//2:]

# t0 = time.time()
# acf_np = acorr(V)
# tf = time.time()
# print(f"calculo con scipy demora: {tf-t0} s")


# teorico:
t_t = np.linspace(0,10,4096)
T = t_t[-1]
acf_teorico = 1/(2*(T-t_t))*(np.exp(-t_t) - np.exp(-(2*T-t_t)))

#%%
# plt.figure(1)
# plt.plot(tau, acf, 'o-', label="manual")
# plt.xlabel("t [s]")
# plt.ylabel("acf")
# plt.legend()
plt.figure(2)
plt.semilogy(tau, acf, 'o', label="numerico")
plt.semilogy(t_t, acf_teorico, label="teorico")
plt.xlabel("t [s]")
plt.ylabel("acf")
plt.legend()
# %%
