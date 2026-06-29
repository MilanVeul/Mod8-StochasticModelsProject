import numpy as np
from typing import Callable


def solve_mgc_waiting_time(lam: float, c: int, E_S: float, F_B: Callable, F_Be: Callable, T_max=60.0, dt=0.05):
    """
    Solves the approximation for W_d using forward step integration.
    """
    rho = (lam * E_S) / c
    if rho >= 1.0:
        raise ValueError("System is unstable (rho >= 1)")
        
    t_grid = np.arange(0, T_max + dt, dt)
    M = len(t_grid)
    W_bar = np.zeros(M)
    
    # Precompute the baseline inhomogeneous term g(t)
    g = (1.0 - rho) * (1.0 - (1.0 - F_Be(t_grid))**c)
    
    # Forward discretization loop
    for i in range(M):
        # Integral term: \lambda * \int_0^{t_i} W_bar(x) * (1 - F(c(t_i - x))) dx
        if i == 0:
            integral = 0.0
        else:
            # Vectorized look-back for x values from t_0 to t_{i-1}
            x_indices = np.arange(i)
            term1 = W_bar[x_indices]
            term2 = 1.0 - F_B(c * (t_grid[i] - t_grid[x_indices]))
            integral = np.sum(term1 * term2) * dt
            
        W_bar[i] = g[i] + lam * integral
        
    return t_grid, W_bar



