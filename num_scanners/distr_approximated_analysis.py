import numpy as np
import matplotlib.pyplot as plt
from waiting_time_approx import solve_mgc_waiting_time

# System Parameters
a, b = 10.0, 19.0
E_S = (a + b) / 2.0 # = 14.5 minutes

# lam = 8.4375 / 60  # Office hours MORNING
# lam = 5.125 / 60  # Office hours AFTERNOON

# lam = 1.234 /60     # Outside office hours

# lam = 3.625 /60     # Weekend days
lam = 1.375 /60     # Weekend night

def F_B(t: np.ndarray):
    """CDF of Uniform[10, 19] service time."""
    return np.clip((t - a) / (b - a), 0.0, 1.0) * (t >= a)

def F_Be(t: np.ndarray):
    """Stationary excess CDF for Uniform[10, 19]."""
    val = np.zeros_like(t)
    # Case 1: t < a
    mask1 = t < a
    val[mask1] = t[mask1]
    # Case 2: a <= t <= b
    mask2 = (t >= a) & (t <= b)
    val[mask2] = a + (b * (t[mask2] - a) - 0.5 * (t[mask2]**2 - a**2)) / (b - a)
    # Case 3: t > b
    mask3 = t > b
    val[mask3] = E_S
    return val / E_S

def plot_distribution(t: np.ndarray, W_conditional: np.ndarray, num_scanners: int):
    plt.figure(figsize=(8, 4))
    plt.plot(t, W_conditional, label=r'$\overline{W(t)}$', color='dodgerblue', lw=2)
    plt.title(f"M/G/c Conditional Waiting Time Distribution (c={num_scanners}, $\\lambda={lam:.2f}$)")
    plt.xlabel("Waiting Time $t$ (minutes)")
    plt.ylabel("Probability")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("diagrams/waiting_time_approximation.pdf")

if __name__ == "__main__":
    allowance = 0.95
    eps = 1e-3
    
    for c in range(1, 5):
        try:
            t, W_conditional = solve_mgc_waiting_time(lam=lam, c=c, E_S=E_S, F_B=F_B, F_Be=F_Be, T_max=70.0, dt=0.01)
            bound_idx = np.where(np.abs(W_conditional - allowance) < eps)[0][0]
            print(f"c={c}: {t[bound_idx]} minutes")
        except:
            print(f"c={c}: unstable")
        

    # plt.figure(figsize=(8, 4))
    # plt.plot(t, W_conditional2, label=r'$c=2$', color='#0072B2', linestyle="-", lw=2)
    # plt.plot(t, W_conditional3, label=r'$c=3$', color='#E69F00', linestyle="--", lw=2)
    # plt.plot(t, W_conditional4, label=r'$c=4$', color='#009E73', linestyle="-.", lw=2)
    # plt.title(f"M/G/c Conditional Waiting Time Distribution")
    # plt.xlabel("Waiting Time $t$ (minutes)")
    # plt.ylabel("Probability")
    # plt.grid(True, linestyle=":", alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("diagrams/waiting_time_approximation.pdf")