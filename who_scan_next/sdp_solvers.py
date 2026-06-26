import numpy as np
from typing import Tuple, Dict, List
import itertools
import random
import matplotlib.pyplot as plt

from sdp_problem import *

# Keys: (n, inpatients, outpatients, emergency) -> Value: float
memo: Dict[Tuple[int, int, int, int], float] = {}

def solve_ovf(n: int, inpatients: int, outpatients: int, emergency: bool, schedule: List[bool]) -> Tuple[float, int]:
    """Returns the optimal value from n to N."""
    # We include schedule in state key to be able to use the memoization table across runs
    state_key = (n, inpatients, outpatients, emergency, tuple(schedule[n:]))
    if state_key in memo: 
        return memo[state_key]
    
    if n == N:
        terminal_penalty = (inpatients * pi_I) + (outpatients * pi_O)
        return -terminal_penalty
    
    # Determine decision space
    d_space = decision_space(inpatients, outpatients, emergency)
    
    opt_value = -float('inf')
    k_values = list(itertools.product([1,0], repeat=3))
    for d in d_space:
        reward = revenue(d) - cost(inpatients, outpatients)
        expected_future = 0.0
        for k in k_values:
            prob = joint_prob(n, k, schedule)
            if prob <= 0.0: continue
            next_I, next_O, next_E = next_state(inpatients, outpatients, emergency, d, k)
            expected_future += prob * solve_ovf(n+1, next_I, next_O, next_E, schedule)
        value = reward + expected_future
        opt_value = max(opt_value, value)

    memo[state_key] = opt_value
    return opt_value

NUM_TESTS = 10
if __name__ == "__main__":
    slot_range = range(N+1)
    # random.seed(77)
    # means = []
    # stds = []

    # for num_taken_slots in slot_range:
    #     print(f"Number of taken slots: {num_taken_slots}")
    #     schedule = [False] * N
    #     for i in range(num_taken_slots):
    #         schedule[i] = True
    #     values = []
    #     for _ in range(NUM_TESTS):
    #         random.shuffle(schedule)
    #         value = solve_ovf(0, 0, 0, 0, schedule)
    #         print(f"  {value}")
    #         values.append(value)
    #     means.append(np.mean(values))
    #     stds.append(np.std(values, ddof=1))
    # print(means)
    # print(stds)
    means = [np.float64(197.0240301271361), np.float64(260.89930508855815), np.float64(350.81126477104243), np.float64(426.96516956771603), np.float64(489.0068413626915), np.float64(552.9241623306754), np.float64(603.9979599720131), np.float64(659.729647761312), np.float64(717.5599874245128), np.float64(765.5126964605995), np.float64(815.9011631256802), np.float64(801.3027730625693), np.float64(839.8993447843588), np.float64(861.7887987563951), np.float64(894.6305191324897), np.float64(890.6803751124933), np.float64(896.133397995907), np.float64(880.7199432408968), np.float64(894.8777895825482), np.float64(921.2821052207257), np.float64(903.7434984278628), np.float64(916.7730204429663), np.float64(902.9511582327004), np.float64(889.145189705487), np.float64(890.0079648809458), np.float64(865.9814512336588), np.float64(859.7076322680111), np.float64(830.2361004338576), np.float64(835.5766332066181), np.float64(842.0450788491905), np.float64(811.4417366966964), np.float64(789.473643962104), np.float64(782.6996706217881)]
    stds = [np.float64(0.0), np.float64(27.80678425989768), np.float64(7.689872992504205), np.float64(15.009378628819602), np.float64(16.702548646115403), np.float64(18.82594440041637), np.float64(43.649389391394244), np.float64(44.382348772628426), np.float64(39.3346426359698), np.float64(39.2734093770565), np.float64(47.192860839878385), np.float64(79.15683361069745), np.float64(39.74638016138413), np.float64(93.93915639199811), np.float64(68.25225765801638), np.float64(89.22746666574307), np.float64(95.44536411988062), np.float64(82.19352344320933), np.float64(75.10687788021146), np.float64(92.15867555699633), np.float64(62.91798224483169), np.float64(77.29906990027904), np.float64(88.12536189143763), np.float64(86.74203585605622), np.float64(64.58005952932234), np.float64(73.69424661363355), np.float64(57.27822203122566), np.float64(51.36423821443198), np.float64(70.12194043900644), np.float64(32.60232124796136), np.float64(30.01912046838805), np.float64(27.302697339199003), np.float64(0.0)]

    # PLOT
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis - Means
    ax1.set_xlabel("Number of Booked Slots ($N_{out}$)", fontsize=14)
    ax1.set_ylabel("Optimal Value Mean", fontsize=14, color='blue')
    ax1.plot(slot_range, means, marker='o', color='blue', label="Expected Value")
    ax1.tick_params(axis='y', labelcolor='blue')

    # Secondary Axis - Variances
    ax2 = ax1.twinx()  
    ax2.set_ylabel("Standard Deviation", fontsize=14, color='crimson')
    ax2.plot(slot_range, stds, marker='o', color='crimson', label="Variance")
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title("Optimal Value Mean with Standard Deviation vs. Number of Booked Slots", fontsize=15)
    
    # Save the figure to your module path directory
    plt.savefig("diagrams/schedule_density_analysis.pdf")