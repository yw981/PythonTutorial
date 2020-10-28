import torch

torch.random.manual_seed(1)
x = torch.rand(2, 5)
print(x)
# scatter_(轴，坐标，数据)，将x中的数值按index数组指定填入y
y = torch.zeros(3, 5).scatter_(0, torch.LongTensor([[2, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)
# 例如，dim = 0
# x = tensor([[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
#         [0.7999, 0.3971, 0.7544, 0.5695, 0.4388]])
# 沿着0轴填写，将x[0,1][0]分别填入y[2,2][0]
# 沿着0轴填写，将x[0,1][1]分别填入y[1,0][1]
# 沿着0轴填写，将x[0,1][2]分别填入y[2,0][2]
# y = tensor([[0.0000, 0.3971, 0.7544, 0.7347, 0.0293],
#         [0.0000, 0.2793, 0.0000, 0.5695, 0.0000],
#         [0.7999, 0.0000, 0.4031, 0.0000, 0.4388]])
print(y)
# 例如，dim = 1
# x = tensor([[0.7576, 0.2793, 0.4031, 0.7347, 0.0293],
#         [0.7999, 0.3971, 0.7544, 0.5695, 0.4388]])
# 沿着1轴填写，将x[0][0,1,2,3,4]分别填入y[0][2,1,2,0,0]，可重复，例如y[0][2]被填了两次，最终0.4031是y[2][2]的值
# 沿着1轴填写，将x[1][0,1,2,3,4]分别填入y[1][2,0,0,1,2]
# y = tensor([[0.0293, 0.2793, 0.4031, 0.0000, 0.0000],
#         [0.7544, 0.5695, 0.4388, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])