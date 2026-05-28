from solvers.value_iteration import discounted_value_iteration, fast_discounted_value_iteration
from typing import List, Tuple
import numpy as np
import scipy.stats as stats

def solve_blueprint_sdp_slow(num_scanners, wait_cost, operating_costs, discount):

    max_state = 500
    states = list(range(max_state+1))
    
    lambda_a = 290

    # precompute distributions
    print("Precomputing distributions...")
    max_arrival = max_state + 32 * (10 * num_scanners+1)
    poisson_pmf = [stats.poisson.pmf(k, lambda_a) for k in range(max_arrival)]
    poisson_cdf = [stats.poisson.cdf(k, lambda_a) for k in range(max_arrival)]

    print("Defining functions...")
    def action_space(i: int) -> List[int]:
        min_d = int(np.ceil(i/32.0))
        max_d = 10 * num_scanners
        if min_d > max_d:
            return [max_d]
        return list(range(min_d, max_d+1))
    
    def costs(i: int, d: int) -> float:
        M = 32*d - i
        assert M >= 0, f"M cannot be negative: M={M}"

        prob_M_min_1 = poisson_cdf[M-1]
        prob_M = poisson_cdf[M]
        expectation = lambda_a * prob_M_min_1 + M * (1 - prob_M)
        exp_cost = (7*i + 2.5*expectation)*wait_cost + d*operating_costs
        return exp_cost
    
    def transitions(i: int, d: int) -> List[Tuple[int, float]]:
        M = 32*d - i
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


def solve_blueprint_sdp_fast(num_scanners, wait_cost, operating_costs, discount):
    max_state = 1000
    epsilon = 1e-2

    max_d = 10 * num_scanners
    num_states = max_state + 1
    num_actions = max_d + 1
    states = list(range(max_state+1))

    lambda_a = 290

    # precompute distributions
    print("Precomputing distributions...")
    max_arrival = max_state + 32*max_d
    # poisson_pmf = [stats.poisson.pmf(k, lambda_a) for k in range(max_arrival)]
    # poisson_cdf = [stats.poisson.cdf(k, lambda_a) for k in range(max_arrival)]
    poisson_pmf = stats.poisson.pmf(np.arange(max_arrival), lambda_a)
    poisson_cdf = stats.poisson.cdf(np.arange(max_arrival), lambda_a)

    
    # Precompute costs and transition matrices
    print("Precomputing transitions and cost matrices")
    costs = np.full((num_states, num_actions), np.inf)              # from,decision   (Invalid actions have infinite costs)
    transitions = np.zeros((num_states, num_actions, num_states))   # from,decision,to

    for i in states:
        min_d = int(np.ceil(i/32.0))
        
        actions = range(min(min_d, max_d), max_d+1)
        for d in actions:
            M = 32*d - i
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
    
    return fast_discounted_value_iteration(
        num_states=num_states,
        costs=costs,
        transitions=transitions,
        discount=discount,
        epsilon=epsilon
    )




if __name__ == "__main__":
    optimal_values, optimal_policy, iterations = solve_blueprint_sdp_fast(num_scanners=20, wait_cost=15, operating_costs=200, discount=0.8)