import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

t = np.linspace(0,10,512)
a1 = 1
a2 = 2
V1 = np.exp(-a1*t)
V2 = np.exp(-a2*t)

VM = np.array([V1,V2])

def acorr(x):
    # result = np.correlate(x, x, mode='full')
    result = signal.correlate(x, x, mode='full')
    return result[result.size//2:]

t0 = time.time()
acf1 = acorr(V1)
acf2 = acorr(V2)
acfM = acorr(VM)
tf = time.time()
print(f"calculo con scipy demora: {tf-t0} s")


# teorico:
t_t = np.linspace(0,10,4096)
T = t_t[-1]
acf_teorico1 = 1/(2*a1)*(np.exp(-a1*t_t) - np.exp(-a1*(2*T-t_t)))
acf_teorico2 = 1/(2*a2)*(np.exp(-a2*t_t) - np.exp(-a2*(2*T-t_t)))
#%%
plt.figure(2)
plt.semilogy(t, acf1/max(acf1), 'o', label="scipy1")
plt.semilogy(t, acf2/max(acf2), 'o', label="scipy2")
plt.semilogy(t_t, acf_teorico1/max(acf_teorico1), label="teorico1")
plt.semilogy(t_t, acf_teorico2/max(acf_teorico2), label="teorico2")

plt.semilogy(t, acfM[:,0]/acfM[:,0], label="scipy-con-array-1")
plt.semilogy(t, acfM[:,1]/acfM[:,1], label="scipy-con-array-2")

plt.xlabel("t [s]")
plt.ylabel("acf")
plt.legend()

# %%
