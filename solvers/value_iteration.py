from typing import Optional, List, Tuple
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

        print(f"Finished iteration {iter:<5} delta={delta}")
        if delta < epsilon:
            break
    else:
        raise Exception("Reached maximum iterations before converging.")
    
    return v, policy


def fast_discounted_value_iteration(
        num_states: List,
        costs: np.ndarray, 
        transitions: np.ndarray,
        discount: float,
        epsilon: float =  1e-6,
        max_iterations: int = 1000
    ):
    v = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    for iter in range(max_iterations):
        expectation = np.dot(transitions, v)
        v_values = costs + discount * expectation

        v_new = np.min(v_values, axis=1)