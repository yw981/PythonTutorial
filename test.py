import numpy as np

a2 = np.array([[55, 66, 77], [11, 22, 33]])
b2 = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
# b4 = np.array([[0, 1],[0,0]])
# b4 = [[0, 1],[0,0]]
# b3 = np.array([[False, True, False], [True, False, False]])
# # print(a2[tuple(b4)])
# print(a2[((0,1),(2,1))])
# print(tuple(np.arange(9)))
# print(a2.shape)
# print(b2.shape)

c = a2.transpose()

print(c.shape)
print(c)
print(a2.shape)
print(a2)

c -= 10
print(c)
print(a2)
