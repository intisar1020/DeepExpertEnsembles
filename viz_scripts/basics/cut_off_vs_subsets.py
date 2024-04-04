import matplotlib.pyplot as plt
import numpy as np

# Create some mock data
t = np.array([1, 2, 3, 5])
# data1 = np.array([731, 107, 50, 28])
# data2 = np.array([755, 194, 88, 32])
pets_sub = np.array([165, 93, 63, 38])
pets_super = np.array([144, 70, 42, 29])
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('cut-off variable')
ax1.set_ylabel('# of supersets (Oxford-IIIT Pet)', color=color)
ax1.plot(t, pets_super, marker='s', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('# of binary sets', color=color)  # we already handled the x-label with ax1
ax2.plot(t, pets_sub, marker='D', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid(linestyle='--', linewidth=1)
plt.show()

# cut-off value for cifar-100
# 506	466	100
# 192	120	99
# 114	56	95
# 62	22	74

# #tiny-imagenet				
# cutoff	binary	super
# 1	755	731
# 2	194	107
# 3	88	50
# 5	32	28

# pets
# cutoff	binary	super	total number of classes
# 1	        165	    144	        37
# 2	        93	    70      	37
# 3	        63	    42	        37
# 5	        38	    29	        36