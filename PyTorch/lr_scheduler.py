import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR

initial_lr = 0.1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


net_1 = Model()

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)

# def lr_func(epoch):
#     return epoch + 1

# LambdaLR学习率变化是关于epoch的函数，可直接用lambda 表达式，更新后的学习率new_lr=λ×initial_lr
# scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1 / (epoch + 1))
# 也可用函数当参数
# scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lr_func)

milestones = [2, 4, 6, 7, 8]
# milestones为一个数组，如 [50,70]. gamma为倍数。每到一个milestone新学习率*gamma，当last_epoch=-1,设定为初始lr。
scheduler_2 = MultiStepLR(optimizer_1, milestones, gamma=0.1)

print("初始化的学习率：", optimizer_1.defaults['lr'])

for epoch in range(0, 11):
    # train
    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    # 用了哪个scheduler在这里体现
    # scheduler_1.step()
    scheduler_2.step()
