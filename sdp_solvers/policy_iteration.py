from typing import List, Tuple, Optional
from collections.abc import Callable
import numpy as np

def policy_iteration(
    num_states: int,
    costs: np.ndarray, 
    transitions: np.ndarray,
    discount: float,
    initial_policy: Optional[np.ndarray] = None,
    max_iterations: int = 1000
):
    
    if initial_policy is None:
        initial_policy = np.zeros(num_states, dtype=int)

    states = np.arange(num_states)
    I = np.eye(num_states)


    policy = initial_policy.astype(int)
    for iter in range(max_iterations):
        # V = C + beta*P*V
        # => (I - discount*P) V = C
        P = transitions[states, policy, :]
        A = I - discount * P
        C = costs[states, policy]
        V = np.linalg.solve(A, C)

        # See value_iteration for clarification
        expectation = np.dot(transitions, V)
        w_values = costs + discount * expectation
        new_policy = np.argmin(w_values, axis=1)

        changed_decisions = np.count_nonzero(policy != new_policy)
        print(f"Iteration {iter:<4} Changed decisions: {changed_decisions}")

        policy = new_policy

        if changed_decisions == 0: # <=> new_policy == policy
            break
    else:
        raise Exception("Reached maximum iterations before converging.")

    v_dict = {i: V[i] for i in range(num_states)}
    policy_dict = {i: policy[i] for i in range(num_states)}
    return v_dict, policy_dict, iter+1

        
