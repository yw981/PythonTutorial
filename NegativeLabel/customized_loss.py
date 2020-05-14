import torch

torch.random.manual_seed(1)
x = torch.rand(2, 5)
print(x)
# scatter_(轴，坐标，数据)
y = torch.zeros(3, 5).scatter_(1, torch.LongTensor([[2, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
print(y)
