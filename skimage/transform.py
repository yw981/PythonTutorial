from skimage import data, filters, transform
import matplotlib.pyplot as plt
import numpy as np

img = data.astronaut()
im1 = img[:, :, 0]
# g1 = filters.gaussian(im1, sigma=5)
# tform = transform.AffineTransform(np.array([[1,0,100],[0,1,100],[0,0,1]]))
# scale : (sx, sy) 缩放，x,y的比例
# tform = transform.AffineTransform(translation=(100, 50)) # 平移（x,y) x正水平向右，y正竖直向上
# tform = transform.AffineTransform(rotation=3.14/10) # 逆时针旋转，弧度，绕左上角顶点
# 剪切变换(shear transformation),方形变平行四边形，任意一边都可以被拉长的过程
tform = transform.AffineTransform(shear=3.14/4)
g1 = transform.warp(im1, tform)
# print(type(img))
# edges1 = filters.gaussian(img,sigma=0) #sigma=0.4
# edges2 = filters.gaussian(img,sigma=5) #sigma=5
# print(edges1.shape)
# print(edges2.shape)
plt.figure('gaussian', figsize=(8, 8))
plt.subplot(121)
plt.imshow(im1, plt.cm.gray)
plt.subplot(122)
plt.imshow(g1, plt.cm.gray)
plt.show()
