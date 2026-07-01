import matplotlib.pyplot as plt
from matplotlib import colors
from when_schedule_outpatients.solve_schedule import greedy_scheduling
from when_schedule_outpatients.analysis import *

max_K = 25
K_values = range(1, max_K+1)
schedule_matrix = []
schedule_matrix = np.zeros((max_K, N), dtype=int)
for idx, k in enumerate(K_values):
    optimal_outpatients, _ = greedy_scheduling(inpatients_per_slot + emergency_per_slot, k, 0.86)
    schedule_matrix[idx, :] = optimal_outpatients

# Create a discrete color mapping since patient counts are integers (0, 1, 2...)
max_allocated = np.max(schedule_matrix)
cmap = plt.cm.get_cmap('Blues', max_allocated + 1)
bounds = np.arange(-0.5, max_allocated + 1.5, 1)
norm = colors.BoundaryNorm(bounds, cmap.N)

# Display the matrix
im = plt.imshow(schedule_matrix, cmap=cmap, norm=norm, aspect='auto',
                extent=[t_start, t_end, max_K + 0.5, 0.5])

# Format Axes
plt.title('Optimal Outpatient Schedule for Each Workload ($K$)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Time of Day (Hours) [15-Minute Slots]', fontsize=10)
plt.ylabel('Outpatients to Schedule ($K$)', fontsize=10)

# X-Axis Ticks (Hourly intervals)
plt.xticks(np.arange(t_start, t_end + 1, 1.0))

# Y-Axis Ticks (Every single value of K)
plt.yticks(K_values)

# Add a distinct vertical line at 12:00 to separate Morning and Afternoon shifts
plt.axvline(x=12.0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='')

# Add discrete colorbar
cbar = plt.colorbar(im, ticks=np.arange(0, max_allocated + 1))
cbar.set_label('Number of Outpatients Scheduled in Slot', fontsize=10)

plt.grid(True, which='both', linestyle=':', alpha=0.3, color='gray')
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig("diagrams/outpatient_schedule_matrix.pdf")