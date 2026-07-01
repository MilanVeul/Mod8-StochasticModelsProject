import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from who_scan_next.sdp_problem import compute_slot_lambdas

# Continuous functions
def lambda_I(t):
    if 9 <= t <= 15:
        return 9 + 72 * (1 + np.cos(2 * np.pi * t / 3 + np.pi))
    return 9
vectorized_lambda = np.vectorize(lambda_I)

N = 32 
t_start, t_end = 8.0, 16.0
L_slot = (t_end - t_start) / N
boundaries = np.linspace(t_start, t_end, N + 1)

# Discrete function
slot_rates_day = []
for i in range(N):
    t_prev, t_curr = boundaries[i], boundaries[i+1]
    integral, _ = quad(lambda_I, t_prev, t_curr)
    Lambda_n = integral / 24 
    avg_rate_day = (Lambda_n * 24) / L_slot
    slot_rates_day.append(avg_rate_day)

# PLOT DISCRETIZED INPATIENT RATE
# t_fine = np.linspace(t_start, t_end, 10000)
# plt.figure(figsize=(8, 4.5))

# plt.plot(t_fine, vectorized_lambda(t_fine), label=r'Continuous $\lambda_I(t)$', color='#1f77b4', lw=2)
# plt.hlines(y=slot_rates_day, xmin=boundaries[:-1], xmax=boundaries[1:], color='#ff7f0e', lw=2.5, label='Discretized Rate')
# plt.axhline(y=21 * 24/8, color='green', linestyle='--', label="Average Rate")

# plt.title('Inpatient Arrival Rates: Continuous vs. Discretized', fontsize=11)
# plt.xlabel('Time of Day (Hours)', fontsize=10)
# plt.ylabel('Arrival Rate (Patients / Day)', fontsize=10)
# plt.xlim(t_start, t_end)
# plt.xticks(np.arange(t_start, t_end + 1, 1.0)) 

# plt.grid(True, linestyle=':', alpha=0.6)
# plt.legend(frameon=True)
# plt.tight_layout()
# plt.savefig("diagrams/discrete_arrival_rates.pdf")