import torch
from torchvision import datasets

# repeat复制tensor的实验
# a= torch.arange(30).reshape(5,6)
# print(a)
# print(a.repeat((3,1,1)))

mnist = datasets.MNIST(root='../../data', train=False)
print(mnist)
