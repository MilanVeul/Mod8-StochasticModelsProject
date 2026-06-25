import scipy.stats as stats
from .dynamic_blueprint import solve_blueprint_sdp_fast, POLICY_ITERATION
import matplotlib.pyplot as plt
import numpy as np
from typing import List


discount = 0.9
lambda_a = 290

wait_costs = 15
operating_costs = 200

num_scanners = 3

def compute_fixed_value_function(I_max: int, d: int):
    states = np.arange(I_max + 1)
    M_vector = 16 * d - states
    
    # Compute immediate costs 
    exp_G_s = lambda_a * stats.poisson.cdf(M_vector - 1, lambda_a) + M_vector * (1 - stats.poisson.cdf(M_vector, lambda_a))
    c_d = (7.0 * states + 2.5 * exp_G_s) * wait_costs + (d * operating_costs)
    
    # Build P_d
    P_d = np.zeros((I_max + 1, I_max + 1))
    for i in range(I_max + 1):
        M = M_vector[i]
        P_d[i, 0] = stats.poisson.cdf(M, lambda_a)
        
        j_overflow = np.arange(1, I_max)
        P_d[i, j_overflow] = stats.poisson.pmf(M + j_overflow, lambda_a)
        P_d[i, I_max] = 1.0 - stats.poisson.cdf(M + I_max - 1, lambda_a)
        
    # Solve (I - beta * P_d) * V_d = c_d
    A = np.eye(I_max + 1) - discount * P_d
    V_d = np.linalg.solve(A, c_d)
    
    return V_d

def compute_optimal_fixed_value_function(d_values: List, max_state: int):
    fixed_value_curves = [compute_fixed_value_function(max_state, d) for d in d_values]
    optimal_fixed_values = np.minimum.reduce(fixed_value_curves)
    return optimal_fixed_values

if __name__ == "__main__":
    max_state = 200
    states = list(range(max_state+1))
    d_values = list(range(int(290 / 16), 10 * num_scanners + 1))

    fixed_values = compute_optimal_fixed_value_function(d_values, max_state)

    dynamic_values_dict, _, _ = solve_blueprint_sdp_fast(POLICY_ITERATION, num_scanners, wait_costs, operating_costs, discount)
    dynamic_values = [dynamic_values_dict[i] for i in states]

    plt.figure(figsize=(10, 6))
    plt.plot(states, fixed_values, label="Optimal Fixed Blueprint (Static $d$)", color="red", linewidth=2)
    plt.plot(states, dynamic_values, label="Optimal Dynamic Blueprint (SDP Policy)", color="dodgerblue", linewidth=2)
    
    plt.title("Fixed vs. Dynamic Blueprint", fontsize=14, fontweight='bold')
    plt.xlabel("Initial Patient Backlog ($i_0$)", fontsize=12)
    plt.ylabel("Expected Long-Term Discounted Cost ($V^*$)", fontsize=12)
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig("diagrams/blueprint-comparison.svg")


    



