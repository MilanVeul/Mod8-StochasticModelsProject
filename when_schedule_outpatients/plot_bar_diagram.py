import matplotlib.pyplot as plt
from when_schedule_outpatients.solve_schedule import greedy_scheduling
from when_schedule_outpatients.analysis import *

K = 15
optimal_outpatients, _ = greedy_scheduling(inpatients_per_slot + emergency_per_slot, K, 0.86)

plt.figure(figsize=(8, 4.5))
slot_centers = boundaries[:-1] + L_slot / 2

# Emergency
plt.bar(slot_centers, emergency_per_slot, width=L_slot * 0.9, 
        color='purple', edgecolor='white', linewidth=0.3, alpha=0.9, 
        label='Expected Emergency Arrivals')
# Outpatients
plt.bar(slot_centers, inpatients_per_slot, width=L_slot * 0.9, 
        bottom=emergency_per_slot, color='#ff7f0e', edgecolor='white', linewidth=0.3, alpha=0.9, 
        label='Expected Inpatient Arrivals')
# Inpatients
plt.bar(slot_centers, optimal_outpatients, width=L_slot * 0.9, 
        bottom=inpatients_per_slot + emergency_per_slot, color='blue', edgecolor='white', linewidth=0.3, alpha=0.8, 
        label='Outpatient Scheduled')

# Line
plt.axvline(x=12.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='')


# Formatting
plt.xlabel('Time of Day (Hours)', fontsize=15)
plt.ylabel('Number of Patients / Slot', fontsize=15) 

plt.xlim(t_start, t_end)
plt.xticks(np.arange(t_start, t_end + 1, 1.0)) 
plt.ylim(0, 2.5)

plt.grid(True, linestyle=':', alpha=0.6, axis='y')
plt.legend(frameon=True, loc='upper right')
plt.tight_layout()

plt.savefig(f"diagrams/outpatient_scheduling/K={K}.pdf")