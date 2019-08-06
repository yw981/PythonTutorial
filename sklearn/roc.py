print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# sklearn官方示例计算并画出ROC曲线
# Import some data to play with
iris = datasets.load_iris()
X = iris.data
# print(X)
y = iris.target
# print(y)

# Binarize the output 标签变成one hot
y = label_binarize(y, classes=[0, 1, 2])
# print(y)
n_classes = y.shape[1]
print(y.shape)

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
# 150个样本，4个特征
n_samples, n_features = X.shape
# 给输入feature加点儿维度，增加难度？
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
print(X.shape)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                    random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                         random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# 预测结果，未归一化，未softmax
print(y_score.shape)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area，Numpy ravel()相当于flatten，内存引用不同
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# tpr fpr都是dict，0,1,2，micro
# print(tpr)
# print(fpr)


plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()