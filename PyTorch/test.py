import torch

# a  = torch.rand(size=(2, 3))

a= torch.arange(30).reshape(5,6)

print(a)

print(a.repeat((3,1,1)))