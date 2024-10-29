import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags

def calculate_acf(process, normalize=True):
    # Calculate the raw ACF
    ACF_raw = correlate(process, process, mode='full')
    ACF_lags = correlation_lags(process.size, process.size, mode='full')

    # Use only positive lags
    half = ACF_raw.size // 2
    ACF_positive = ACF_raw[half:]
    ACF_lags_positive = ACF_lags[half:]

    if normalize:
        # Edge effect normalization
        Ntau = ACF_lags_positive.size
        normalization = Ntau - np.arange(Ntau)
        ACF_positive = ACF_positive / normalization
        # Normalize by zero-lag value for true ACF
        ACF_positive = ACF_positive / ACF_positive[0]

    return ACF_lags_positive, ACF_positive

# Generate each process

# 1. White Noise
np.random.seed(0)
n_points = 1000
white_noise = np.random.random(n_points)
# white_noise = np.random.normal(0, 1, n_points)

# 2. Brownian Motion (Random Walk)
brownian_motion = np.cumsum(white_noise)

# 3. Ornstein-Uhlenbeck Process
theta = 0.15   # Mean reversion rate
mu = 0.0       # Long-term mean
sigma = 0.3    # Volatility
dt = 0.01      # Time step

ou_process = np.zeros(n_points)
for t in range(1, n_points):
    ou_process[t] = (ou_process[t-1] 
                     + theta * (mu - ou_process[t-1]) * dt 
                     + sigma * np.sqrt(dt) * np.random.normal())

# Calculate ACFs
lags_wn, acf_wn_norm = calculate_acf(white_noise, normalize=True)
_, acf_wn_no_norm = calculate_acf(white_noise, normalize=False)

lags_bm, acf_bm_norm = calculate_acf(brownian_motion, normalize=True)
_, acf_bm_no_norm = calculate_acf(brownian_motion, normalize=False)

lags_ou, acf_ou_norm = calculate_acf(ou_process, normalize=True)
_, acf_ou_no_norm = calculate_acf(ou_process, normalize=False)

# Plot the signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(white_noise, color='blue')
plt.title("White Noise Signal")
plt.xlabel("Time step")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(brownian_motion, color='green')
plt.title("Brownian Motion Signal")
plt.xlabel("Time step")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(ou_process, color='purple')
plt.title("Ornstein-Uhlenbeck Process Signal")
plt.xlabel("Time step")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Plot the ACF comparisons
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(lags_wn, acf_wn_norm/max(acf_wn_norm), label='Normalized ACF', color='blue')
plt.plot(lags_wn, acf_wn_no_norm/max(acf_wn_no_norm), '--', label='Raw ACF', color='cyan')
plt.title("White Noise ACF Comparison")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(lags_bm, acf_bm_norm/max(acf_bm_norm), label='Normalized ACF', color='green')
plt.plot(lags_bm, acf_bm_no_norm/max(acf_bm_no_norm), '--', label='Raw ACF', color='lime')
plt.title("Brownian Motion ACF Comparison")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(lags_ou, acf_ou_norm/max(acf_ou_norm), label='Normalized ACF', color='purple')
plt.plot(lags_ou, acf_ou_no_norm/max(acf_ou_no_norm), '--', label='Raw ACF', color='violet')
plt.title("Ornstein-Uhlenbeck Process ACF Comparison")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()

plt.tight_layout()
plt.show()





