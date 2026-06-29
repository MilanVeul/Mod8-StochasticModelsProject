import numpy as np
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid

# CT: 0, service desk: 1, anesthetist: 2, CPM: 3, Additional tests: 4
arrival_rates = np.array([8.4375, 8.3, 7.43727599, 3.63682796, 1.62132616])

Tmax = 60
dt = 0.05

t_grid = np.arange(0, Tmax + dt, dt)

############# DISTRIBUTIONS ##################
ct_distr = stats.uniform(loc=10, scale=9) # [10, 19]
secr_distr = stats.gamma(a=1.01, scale=2.6931)
anes_distr = stats.gamma(a=1.03, scale=18.7736)
cpm_distr  = stats.gamma(a=1.02, scale=24.7861)

mu = 10.27
sigma = np.sqrt(5.617)
a_std = -mu / sigma
b_std = np.inf
test_distr  = stats.truncnorm(a=a_std, b=b_std, loc=10.27, scale=sigma)

def solve_approximation(lam_hr, c, dist, t_grid):
    dt = t_grid[1] - t_grid[0]
    N = len(t_grid)
    
    lam = lam_hr / 60.0
    rho = (lam * dist.mean()) / c
    if rho >= 1.0:
        raise ValueError(f"System unstable (rho = {rho:.4f} >= 1)")
        
    # 2. Compute F_Be(t)
    f_Be = dist.sf(t_grid) / dist.mean()
    F_Be = cumulative_trapezoid(f_Be, t_grid, initial=0)
    
    g = (1.0 - rho) * (1.0 - (1.0 - F_Be)**c)
    
    K_grid = dist.sf(c * t_grid)
    
    # Forward Step Integration
    Wd_bar = np.zeros(N)
    Wd_bar[0] = g[0] # Evaluates to 0 at t=0
    
    denom = 1.0 - lam * (dt / 2.0)
    
    for i in range(1, N):
        sum_terms = np.sum(Wd_bar[1:i] * K_grid[i-1:0:-1])
        boundary = (dt / 2.0) * Wd_bar[0] * K_grid[i]
        
        # Compute forward step
        num = g[i] + lam * (boundary + dt * sum_terms)
        Wd_bar[i] = num / denom
        
    return Wd_bar

Wd_bar_ct = solve_approximation(lam_hr=arrival_rates[0], c=3, dist=ct_distr, t_grid=t_grid)
Wd_bar_secr = solve_approximation(lam_hr=arrival_rates[1], c=1, dist=secr_distr, t_grid=t_grid)
Wd_bar_anes = solve_approximation(lam_hr=arrival_rates[2], c=3, dist=anes_distr, t_grid=t_grid)
Wd_bar_cpm = solve_approximation(lam_hr=arrival_rates[3], c=2, dist=cpm_distr, t_grid=t_grid)
Wd_bar_test = solve_approximation(lam_hr=arrival_rates[4], c=2, dist=test_distr, t_grid=t_grid)
