import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
class_count = np.array([59, 122, 175, 200])
#data1 = np.array([466, 120, 56, 22])
subsets_ = np.array([28, 50, 107, 731])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('# of subsets')
ax1.plot(subsets_, class_count, color=color)

ax1.set_ylabel('class count', color=color)
ax1.tick_params(axis='y', labelcolor=color)

#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('# of binary sets', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid(linestyle='--', linewidth=1)
plt.show()

# TinyImageNet
# super no. of cls
# 107	175
# 50	122
# 28	59