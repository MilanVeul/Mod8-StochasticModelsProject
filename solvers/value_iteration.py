from typing import List, Tuple
from collections.abc import Callable
import numpy as np

def discounted_value_iteration(
        states: List,
        action_space: Callable[[int], List],
        costs: Callable[[int, int], float], 
        transitions: Callable[[int, int], Tuple[int, float]],
        discount: float,
        epsilon: float =  1e-6,
        max_iterations: int = 1000
    ):

    v = {s: 0.0 for s in states}
    policy = {s: None for s in states}
    for iter in range(max_iterations):
        delta = 0
        v_new = v.copy()

        for i in states:
            actions = action_space(i)
            if not actions:
                continue

            action_values = []
            for d in actions:
                c = costs(i, d)

                expectation = sum(
                    prob * v[target]
                    for target, prob in transitions(i, d))

                value = c + discount * expectation

                action_values.append((value, d))
            best_value, best_action = min(action_values, key=lambda x: x[0])

            v_new[i] = best_value
            policy[i] = best_action

            delta = max(delta, abs(v_new[i] - v[i]))
        v = v_new

        print(f"Iteration {iter:<4} Delta: {delta:.6e}")

        if delta < epsilon:
            break
    else:
        raise Exception("Reached maximum iterations before converging.")
    
    return v, policy, iter


def fast_discounted_value_iteration(
        num_states: int,
        costs: np.ndarray, 
        transitions: np.ndarray,
        discount: float,
        epsilon: float =  1e-6,
        max_iterations: int = 1000
    ) -> Tuple[dict[int, float], dict[int, int], int]:
    # Initiate values and policy
    v = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    for iter in range(max_iterations):
        # Docs: "If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b"
        # so this is a dot product of probabilities col and value col
        expectation = np.dot(transitions, v)        # 2d array
        v_values = costs + discount * expectation   # 2d array (rows are the decisions, columns are states)

        v_new = np.min(v_values, axis=1)            # 1d array (get the minimum of over columns (=decisions))
        delta = np.max(np.abs(v_new - v))           # 0d array (Single value)

        v = v_new

        converged = delta < epsilon

        if iter % 10 == 0 or converged:
            print(f"Iteration {iter:<4} Delta: {delta:.6e}")

        if converged:
            policy = np.argmin(v_values, axis=1)    # Get decisions (row) that minimizes the columns
            break
    else:
        raise Exception("Reached maximum iterations before converging.")
    
    # Construct easy-to-use dicts
    v_dict = {i: v[i] for i in range(num_states)}
    policy_dict = {i: policy[i] for i in range(num_states)}
    return v_dict, policy_dict, iter
