import matplotlib.pyplot as plt
import numpy as np

# 画图查看
file_path = 'aa_lenet_mnist_fsgm.npy'
images = np.load(file_path)

cnt = 0
row = 8
col = 16
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
