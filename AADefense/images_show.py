import matplotlib.pyplot as plt
import numpy as np

# 画图查看
# file_path_prefix = 'result/aa_lenet_mnist_fsgm'
file_path_prefix = 'result/aa_lenet_mnist_cw'
file_path_data = file_path_prefix + '.npy'
images = np.load(file_path_data)

cnt = 0
row = 6
col = 8
plt.figure(figsize=(col, row))
for i in range(row):
    for j in range(col):
        cnt += 1
        plt.subplot(row, col, cnt)
        plt.xticks([], [])
        plt.yticks([], [])

        ex = np.squeeze(images[i * col + j])
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
