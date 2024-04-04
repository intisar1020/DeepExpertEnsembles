import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel('no. of binary subsets.')
ax.set_ylabel('accuracy (%)')
# following dataset is for c-100 dataset with ResNet20
# ICC based
acc = [69.58, 69.80, 70.43, 71.16, 71.15, 72.50, 72.81]
subset = [0, 1, 3, 5, 10, 30, 50]
#ax1.plot(cut_off, router, color=color)
ax.plot(subset, acc, linestyle='dashed', linewidth=1, marker='s')
plt.ylim(69.5, 73)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid(linestyle='--', linewidth=1)
#plt.legend()
plt.show()