from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
cut_off = np.array([2, 3, 5])
pets_2 = np.array([63.4, 75.46, 73.0])
pets_3 = np.array([63.4, 68.30, 67.00])
pets_5 = np.array([76.17, 75.46, 73.0])
router = np.array([63.4, 63.4, 63.4])
ensemble = np.array([67.95, 67.95, 67.95])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('cut-off variable')
ax1.set_ylabel('accuracy (%)', color=color)
#ax1.plot(cut_off, router, color=color)
ax1.plot(cut_off, router, color='green', linestyle='dashed',
     linewidth=1, label="router")

ax1.plot(cut_off, ensemble, color='red', linestyle='dashed',
     linewidth=1, label="ensemblex4")
ax1.plot(cut_off, pets_3, marker="s", label='experts')
ax1.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid(linestyle='--', linewidth=1)
plt.legend()
plt.show()