import numpy as np

def greedy_scheduling(P, K, p_s=1.0):
    """
    Allocates K outpatients across 32 slots optimizing day-part variance 
    under strict shift constraints (Max 18 morning, Max 7 afternoon).
    """
    N = len(P)
    S = np.zeros(N, dtype=int)
    A = P.copy()
    
    k_morn = 0
    k_aft = 0
    
    # Internal function to compute exact regional variance/MSE objective value
    def compute_objective(arrival_profile):
        morn = arrival_profile[:16]
        aft = arrival_profile[16:]
        return np.sum((morn - np.mean(morn))**2) + np.sum((aft - np.mean(aft))**2)
    
    for k in range(K):
        # Identify the deepest baseline valleys inside each respective day-part
        i_morn = np.argmin(A[:16])
        i_aft = 16 + np.argmin(A[16:])
        
        # Scenario A: Both shifts still have open capacity
        if k_morn < 18 and k_aft < 7:
            # Look ahead: Simulate putting the patient in the morning valley
            A_morn_test = A.copy()
            A_morn_test[i_morn] += p_s
            obj_morn = compute_objective(A_morn_test)
            
            # Look ahead: Simulate putting the patient in the afternoon valley
            A_aft_test = A.copy()
            A_aft_test[i_aft] += p_s
            obj_aft = compute_objective(A_aft_test)
            
            # Choose the shift that yields the smaller objective value
            if obj_morn <= obj_aft:
                i_star = i_morn
                k_morn += 1
            else:
                i_star = i_aft
                k_aft += 1
                
        # Scenario B: Afternoon shift is filled up to capacity, force Morning
        elif k_morn < 18:
            i_star = i_morn
            k_morn += 1
            
        # Scenario C: Morning shift is filled up to capacity, force Afternoon
        elif k_aft < 7:
            i_star = i_aft
            k_aft += 1
        else: 
            raise ValueError()
            
        # Commit the assignment
        S[i_star] += 1
        A[i_star] += p_s
        
    return S, A