from sdp_solvers.value_iteration import discounted_value_iteration, fast_discounted_value_iteration
from sdp_solvers.policy_iteration import policy_iteration
from typing import List, Tuple
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

VALUE_ITERATION = "Value Iteration"
POLICY_ITERATION = "Policy Iteration"

def solve_blueprint_sdp_slow(num_scanners, wait_cost, operating_costs, discount):

    max_state = 500
    states = list(range(max_state+1))
    
    lambda_a = 290

    # precompute distributions
    print("Precomputing distributions...")
    max_arrival = max_state + 16 * (10 * num_scanners+1)
    poisson_pmf = [stats.poisson.pmf(k, lambda_a) for k in range(max_arrival)]
    poisson_cdf = [stats.poisson.cdf(k, lambda_a) for k in range(max_arrival)]

    print("Defining functions...")
    def action_space(i: int) -> List[int]:
        min_d = int(np.ceil(i/16.0))
        max_d = 10 * num_scanners
        if min_d > max_d:
            return [max_d]
        return list(range(min_d, max_d+1))
    
    def costs(i: int, d: int) -> float:
        M = 16*d - i
        assert M >= 0, f"M cannot be negative: M={M}"

        prob_M_min_1 = poisson_cdf[M-1]
        prob_M = poisson_cdf[M]
        expectation = lambda_a * prob_M_min_1 + M * (1 - prob_M)
        exp_cost = (7*i + 2.5*expectation)*wait_cost + d*operating_costs
        return exp_cost
    
    def transitions(i: int, d: int) -> List[Tuple[int, float]]:
        M = 16*d - i
        assert M >= 0, f"M cannot be negative: M={M}"
        trans_list = []

        prob_zero = poisson_cdf[M]
        trans_list.append((0, prob_zero))

        for j in states:
            prob_j = poisson_pmf[M+j]
            if prob_j < 1e-12: continue # Negligible
            trans_list.append((j, prob_j))
        prob_tail = 1 - poisson_cdf[M + max_state - 1]
        trans_list.append((max_state, prob_tail))
        return trans_list

    return discounted_value_iteration(
        states=states,
        action_space=action_space,
        costs=costs,
        transitions=transitions,
        discount=discount,
        epsilon=1.0
    )


def solve_blueprint_sdp_fast(solver_type, num_scanners, wait_cost, operating_costs, discount) -> Tuple[dict, dict, int]:
    max_state = 500
    epsilon = 1e-2

    max_d = 10 * num_scanners
    num_states = max_state + 1
    num_actions = max_d + 1
    states = list(range(max_state+1))

    lambda_a = 290

    # precompute distributions
    print("Precomputing distributions...")
    max_arrival = max_state + 16*max_d
    # poisson_pmf = [stats.poisson.pmf(k, lambda_a) for k in range(max_arrival)]
    # poisson_cdf = [stats.poisson.cdf(k, lambda_a) for k in range(max_arrival)]
    poisson_pmf = stats.poisson.pmf(np.arange(max_arrival), lambda_a)
    poisson_cdf = stats.poisson.cdf(np.arange(max_arrival), lambda_a)

    
    # Precompute costs and transition matrices
    print("Precomputing transitions and cost matrices")
    costs = np.full((num_states, num_actions), np.inf)              # from,decision   (Invalid actions have infinite costs)
    transitions = np.zeros((num_states, num_actions, num_states))   # from,decision,to

    valid_init_policy = np.zeros(num_states, dtype=int)

    for i in states:
        min_d = int(np.ceil(i/16.0))
        valid_init_policy[i] = min(min_d, max_d)
        
        actions = range(min(min_d, max_d), max_d+1)
        for d in actions:
            M = 16*d - i
            assert M >= 0, f"M cannot be negative: M={M}"

            # Compute costs
            prob_M_min_1 = poisson_cdf[M-1]
            prob_M = poisson_cdf[M]
            expectation = lambda_a * prob_M_min_1 + M * (1 - prob_M)
            costs[i, d] = (7*i + 2.5*expectation)*wait_cost + d*operating_costs

            # Compute transitions
            transitions[i, d, 0] = poisson_cdf[M]
            j_vals = np.arange(1, max_state)
            transitions[i, d, j_vals] = poisson_pmf[M + j_vals]
            transitions[i, d, max_state] = 1 - poisson_cdf[M + max_state - 1]
    
    print("Starting iteration...")
    if solver_type == VALUE_ITERATION:
        return fast_discounted_value_iteration(
            num_states=num_states,
            costs=costs,
            transitions=transitions,
            discount=discount,
            epsilon=epsilon
        )
    
    if solver_type == POLICY_ITERATION:
        return policy_iteration(
            num_states=num_states,
            costs=costs,
            transitions=transitions,
            discount=discount,
            initial_policy=valid_init_policy
        )

# Written by AI
def plot_solution(solver_type: str, optimal_values: dict, optimal_policy: dict):
    sorted_states = sorted(optimal_values.keys())
    
    states_list = [int(s) for s in sorted_states]
    values_list = [float(optimal_values[s]) for s in sorted_states]
    policy_list = [int(optimal_policy[s]) for s in sorted_states]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Optimal Dynamic Blueprint", fontsize=14, fontweight='bold')

    # Left Subplot: Optimal Capacity Decisions (Policy)
    ax1.plot(states_list, policy_list, color='darkblue', linewidth=2, label="Optimal Policy ($d^*$)")
    ax1.set_title("Optimal Actions (Scanner Day-Parts to Open)")
    ax1.set_xlabel("Initial Patient Backlog ($i$)")
    ax1.set_ylabel("Number of Day-Parts ($d$)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Right Subplot: Value Function (Long-Term Expected Discounted Cost)
    ax2.plot(states_list, values_list, color='crimson', linewidth=2, label="Value Function ($V^*$)")
    ax2.set_title("Expected Value Function (Total Discounted Cost)")
    ax2.set_xlabel("Initial Patient Backlog ($i$)")
    ax2.set_ylabel("Expected Long-Term Cost")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Adjust layout and display the figures
    plt.tight_layout()
    plt.savefig("diagrams/dynamic-blueprint-solution.png")

def print_solution(solver_type: str, optimal_values: dict, optimal_policy: dict, iterations: int, limit: int = -1):
    print("\n" + "="*50)
    print(f"  {solver_type} converged in {iterations} iterations.")
    print(f"{'Patient':<9} | {'Decision':<10} | {'Expected Cost (V)':<15}")
    print("-"*55)
    
    max = limit
    if limit == -1:
        max = len(optimal_values)

    for state in range(max):
        cost = optimal_values[state]
        action = optimal_policy[state]
        print(f"{state:<9} | {action:<10} | {cost:<15.2f}")
        
    print("...")


if __name__ == "__main__":
    optimal_values, optimal_policy, iterations = solve_blueprint_sdp_fast(POLICY_ITERATION, num_scanners=3, wait_cost=15, operating_costs=200, discount=0.9)
    plot_solution(POLICY_ITERATION, optimal_values, optimal_policy)
    # print_solution(POLICY_ITERATION, optimal_values, optimal_policy, iterations, 20)