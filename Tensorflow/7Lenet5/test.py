import numpy as np

r2 = np.loadtxt('vy_pred_right2.npy')
c1, c2 = np.split(r2, 2, axis=1)
ratio_r = c1 / c2
np.savetxt('ratio_right', ratio_r)
print('ratio_right mean ', np.min(ratio_r), np.max(ratio_r), np.mean(ratio_r), np.var(ratio_r))
print('it ', np.sum(ratio_r < 14))

w2 = np.loadtxt('vy_pred_wrong2.npy')
c1, c2 = np.split(w2, 2, axis=1)
ratio_w = c1 / c2
np.savetxt('ratio_wrong', ratio_w)
print('ratio_wrong mean ', np.min(ratio_w), np.max(ratio_w), np.mean(ratio_w), np.var(ratio_w))
print('it ', np.sum(ratio_w < 14))

for i in range(1, 200):
    print(np.sum(ratio_r < i), np.sum(ratio_w < i))
