import numpy as np
from base import img_set
from base import model_name
from base import target_id

filename = 'result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id)
data = np.load(filename)

# ---------- norm data 数据预处理，如不预处理导致性能下降和攻击失败-----------
norm_data = data.transpose((2, 3, 0, 1))
# 上一行！！注意numpy的赋值是浅拷贝，赋值后引用的仍是原位置，改变后会影响原变量，深拷贝b = a.copy()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# 必须和最末维度相同才能广播，由于是浅拷贝，data值也跟着改变了
norm_data -= mean
norm_data /= std
norm_data = norm_data.transpose((2, 3, 0, 1))

np.save(filename[0:filename.find('.')] + 'norm.npy', norm_data)
print('file saved ', filename[0:filename.find('.')] + 'norm.npy')