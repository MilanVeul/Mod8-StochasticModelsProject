import numpy as np
import scipy.stats as stats
from scipy.integrate import cumulative_trapezoid, trapezoid

# CT: 0, secretary: 1, anesthetist: 2, CPM: 3, Additional tests: 4
arrival_rates = np.array([8.4375, 8.3, 8.3, 2.75975, 1.826])

Tmax = 1000
dt = 0.05

t_grid = np.arange(0, Tmax + dt, dt)

############# DISTRIBUTIONS ##################
print("Computing distributions...")
ct_distr = stats.uniform(loc=10, scale=9) # [10, 19]
secr_distr = stats.gamma(a=1.01, scale=2.6931)
anes_distr = stats.gamma(a=1.03, scale=18.7736)
cpm_distr  = stats.gamma(a=1.02, scale=24.7861)

mu = 10.27
sigma = np.sqrt(5.617)
a_std = -mu / sigma
b_std = np.inf
test_distr  = stats.truncnorm(a=a_std, b=b_std, loc=10.27, scale=sigma)

def compute_Wd(lam_hr, c, dist, t_grid):
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
    Wd = np.zeros(N)
    Wd[0] = g[0] # Evaluates to 0 at t=0
    
    denom = 1.0 - lam * (dt / 2.0)
    
    for i in range(1, N):
        sum_terms = np.sum(Wd[1:i] * K_grid[i-1:0:-1])
        boundary = (dt / 2.0) * Wd[0] * K_grid[i]
        # Compute forward step
        num = g[i] + lam * (boundary + dt * sum_terms)
        Wd[i] = num / denom
        
    return Wd

print("Computing waiting times...")
Wd_ct = compute_Wd(lam_hr=arrival_rates[0], c=3, dist=ct_distr, t_grid=t_grid)
Wd_secr = compute_Wd(lam_hr=arrival_rates[1], c=1, dist=secr_distr, t_grid=t_grid)
Wd_anes = compute_Wd(lam_hr=arrival_rates[2], c=3, dist=anes_distr, t_grid=t_grid)
Wd_cpm = compute_Wd(lam_hr=arrival_rates[3], c=2, dist=cpm_distr, t_grid=t_grid)
Wd_test = compute_Wd(lam_hr=arrival_rates[4], c=2, dist=test_distr, t_grid=t_grid)

def compute_Ld(Wd: np.ndarray, t_grid: np.ndarray, lam_hr: float, K_max: int = 100):
    lam = lam_hr / 60.0
    L_d = np.zeros(K_max + 1)

    # Case k = 0
    integrand = Wd * np.exp(-lam*t_grid)
    L_d[0] = lam * trapezoid(integrand, t_grid)
    
    k_values = np.arange(1, K_max+1)
    
    # Case K >= 1
    for k in k_values:
        pois_min1 = stats.poisson.pmf(k-1, lam * t_grid)
        pois = stats.poisson.pmf(k, lam * t_grid)
        boundary = stats.poisson.pmf(k, lam * Tmax) * Wd[-1]
        integrand = Wd * (pois - pois_min1)
        L_d[k] = boundary + lam * trapezoid(integrand, t_grid)
    return L_d

print("Computing queue lengths...")
K_max = 100
Ld_ct = compute_Ld(Wd=Wd_ct, t_grid=t_grid, lam_hr=arrival_rates[0], K_max=K_max)
Ld_secr = compute_Ld(Wd=Wd_secr, t_grid=t_grid, lam_hr=arrival_rates[1], K_max=K_max)
Ld_anes = compute_Ld(Wd=Wd_anes, t_grid=t_grid, lam_hr=arrival_rates[2], K_max=K_max)
Ld_cpm = compute_Ld(Wd=Wd_cpm, t_grid=t_grid, lam_hr=arrival_rates[3], K_max=K_max)
Ld_test = compute_Ld(Wd=Wd_test, t_grid=t_grid, lam_hr=arrival_rates[4], K_max=K_max)

print("Computing joint probability...")
joint_prob = Ld_ct
joint_prob = np.convolve(joint_prob, Ld_secr)
joint_prob = np.convolve(joint_prob, Ld_anes)
joint_prob = np.convolve(joint_prob, Ld_cpm)
joint_prob = np.convolve(joint_prob, Ld_test)

cum_joint_prob = np.cumsum(joint_prob)

for k in [41, 42, 43, 44, 45]:
    print(f"P(L_q > {k}) = {1-cum_joint_prob[k]}")

print(f"Required chairs: {np.searchsorted(cum_joint_prob, 0.99)}")