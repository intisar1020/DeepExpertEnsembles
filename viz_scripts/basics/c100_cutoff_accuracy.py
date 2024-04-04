from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
cut_off = np.array([2, 3, 5])
cifar100_2 = np.array([75.21, 75.46, 73.0])
cifar100_3 = np.array([76.25, 75.46, 73.0])
cifar100_5 = np.array([76.17, 75.46, 73.0])
router = np.array([70.78, 70.78, 70.78])
ensemble = np.array([75.0, 75.0, 75.0])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('cut-off variable')
ax1.set_ylabel('accuracy (%)', color=color)
#ax1.plot(cut_off, router, color=color)
ax1.plot(cut_off, router, color='green', linestyle='dashed',
     linewidth=1, label="router")

ax1.plot(cut_off, ensemble, color='red', linestyle='dashed',
     linewidth=1, label="ensemblex4")
ax1.plot(cut_off, cifar100_3, marker="s", label='experts')
ax1.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid(linestyle='--', linewidth=1)
plt.legend()
plt.show()