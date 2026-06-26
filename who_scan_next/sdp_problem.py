import numpy as np
from scipy.integrate import quad
from typing import Tuple, Dict, List
from enum import Enum

class Decision(Enum):
    INPATIENT = 0
    OUTPATIENT = 1
    EMERGENCY = 2
    IDLE = 3

def lambda_I(t: float) -> float:
    # Ensure t wraps around 24 hours just in case
    t_mod = t % 24.0
    
    if 9.0 <= t_mod <= 15.0:
        return 9.0 + 72.0 * (1.0 + np.cos((2.0 * np.pi * t_mod) / 3.0 + np.pi))
    else:
        return 9.0

def compute_slot_lambdas(N: int) -> np.ndarray:
    """
    Computes the average arrival rate lambda_n for each of the N slots during office hours (t in [8, 16]).
    """
    L_slot = 8.0 / N
    lambda_n_list = []
    
    start_office = 8.0
    
    for n in range(1, N + 1):
        t_start = start_office + (n - 1) * L_slot
        t_end = start_office + n * L_slot
        integral_val, _ = quad(lambda_I, t_start, t_end)
        lambda_n = (1.0 / 24) * integral_val
        lambda_n_list.append(lambda_n)
    return np.array(lambda_n_list)

N = 2*16
p_emergency = 0.221
p_show = 0.86
lambda_n = compute_slot_lambdas(N)

r_O = 100
r_I = 20
w_O = 1.5
w_I = 0
pi_O = 10
pi_I = 200

def cost(inpatients: int, outpatients: int) -> float:
    return w_I + inpatients + w_O * outpatients
def revenue(decision: int) -> float: 
    if decision == Decision.INPATIENT:
        return r_I
    if decision == Decision.OUTPATIENT: 
        return r_O
    return 0.0


def joint_prob(n: int, k: Tuple[bool, bool, bool], schedule: List[bool]) -> float:
    k_I, k_O, k_E = k
    if n > 1:   p_I = 1-np.exp(-lambda_n[n])
    else:       p_I = 0
    p_O = p_show * schedule[n]
    p_E = p_emergency
    p = 1
    if k_I: p *= p_I
    else:   p *= 1-p_I
    if k_O: p *= p_O
    else:   p *= 1-p_O
    if k_E: p *= p_E
    else:   p *= 1-p_E
    return p

def decision_space(I: int, O: int, E: bool) -> List[Decision]:
    if E == 1:
        return [Decision.EMERGENCY]
    
    d_space = [Decision.IDLE]
    if I > 0: d_space.append(Decision.INPATIENT)
    if O > 0: d_space.append(Decision.OUTPATIENT)
    return d_space

def next_state(I: int, O: int, E: int, d: Decision, k: Tuple[bool, bool, bool]) -> Tuple[int, int, bool]:
    k_I, k_O, k_E = k
    next_I = I + k_I
    next_O = O + k_O
    next_E = k_E
    if d == Decision.INPATIENT: next_I -= 1
    if d == Decision.OUTPATIENT: next_O -= 1
    return (next_I, next_O, next_E)