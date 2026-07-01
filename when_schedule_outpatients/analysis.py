import numpy as np

from scipy.integrate import quad
from scipy.optimize import minimize

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
slot_rates_day = np.array(slot_rates_day)
inpatients_per_slot = (slot_rates_day/3) / N

# Emergency patients
emergency_per_slot = np.array([8 / N] * N)
