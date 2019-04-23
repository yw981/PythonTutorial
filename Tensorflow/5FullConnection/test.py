import numpy as np
import matplotlib.pyplot as plt

r = np.loadtxt('vy_err_right.npy')
w = np.loadtxt('vy_err_wrong.npy')

bins = np.linspace(0, 0.8, 100)
plt.figure()
plt.hist(r, bins)
plt.figure()
plt.hist(w, bins, color='orange')

bins1 = np.linspace(0, 0.04, 100)
plt.figure()
plt.hist(r, bins1, color='green')
plt.figure()
plt.hist(w, bins1, color='yellow')
plt.show()
