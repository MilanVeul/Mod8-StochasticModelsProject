import scipy.stats as stats
from .dynamic_blueprint import solve_blueprint_sdp_fast, POLICY_ITERATION
import matplotlib.pyplot as plt
import numpy as np


discount = 0.9
lambda_a = 290

wait_costs = 15
operating_costs = 200

num_scanners = 3


m_min, m_max = -600, 1000  
m_range = np.arange(m_min, m_max + 1)
# Precompute cdfs
cdf_M_minus_1 = stats.poisson.cdf(m_range - 1, lambda_a)
cdf_M = stats.poisson.cdf(m_range, lambda_a)
exp_values = lambda_a * cdf_M_minus_1 + m_range * (1 - cdf_M)
exp_lookup = {m: val for m, val in zip(m_range, exp_values)}

def compute_fixed_value(initial_state: int, d: int) -> float:
    state = float(initial_state)
    discounted_cost = 0.0
    current_discount = 1.0
    epsilon = 1e-6
    
    # Pre-extract values outside loop
    d_op_cost = d * operating_costs

    while current_discount > epsilon:
        # Fast lookup with a fallback if M drifts outside precomputed limits
        M = int(16*d - round(state))
        expectation = exp_lookup.get(M, lambda_a if M < 0 else 0.0)

        cost = (7.0 * state + 2.5 * expectation) * wait_costs + d_op_cost
        discounted_cost += current_discount * cost

        current_discount *= discount
        state = lambda_a - expectation
    
    return discounted_cost

def compute_optimal_value(initial_state: int):
    print(f"Computing state {initial_state}")
    d_values = range(int(290/16), 10*num_scanners + 1)
    minimum_value = 1e9
    minimum_d = -1
    for d in d_values:
        value = compute_fixed_value(initial_state, d)
        if value < minimum_value:
            minimum_value = value
            minimum_d = d
    return minimum_value, minimum_d


if __name__ == "__main__":
    max_state = 200
    states = list(range(max_state+1))
    d_values = list(range(int(290 / 16), 10 * num_scanners + 1))

    fixed_values = []
    for init_state in states:
        min_value = 1e9
        for d in d_values:
            if 16 * d < init_state: 
                continue
            value = compute_fixed_value(init_state, d)
            if value < min_value:
                min_value = value
        fixed_values.append(min_value)


    dynamic_values_dict, _, _ = solve_blueprint_sdp_fast(POLICY_ITERATION, num_scanners, wait_costs, operating_costs, discount)
    dynamic_values = [dynamic_values_dict[i] for i in states]

    plt.figure(figsize=(10, 6))
    plt.plot(states, fixed_values, label="Optimal Fixed Blueprint (Static $d$)", color="crimson", linewidth=2)
    plt.plot(states, dynamic_values, label="Optimal Dynamic Blueprint (SDP Policy)", color="dodgerblue", linewidth=2, linestyle="--")
    
    plt.title("CoFixed vs. Dynamic Blueprint", fontsize=14, fontweight='bold')
    plt.xlabel("Initial Patient Backlog ($i$)", fontsize=12)
    plt.ylabel("Expected Long-Term Discounted Cost ($V^*$)", fontsize=12)
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig("diagrams/blueprint-comparison.png")


    



