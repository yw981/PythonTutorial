import torch

a = torch.tensor(range(12)).float().view(-1, 4)
print(a.size())
print(a)
# exit(0)

# sm0 = torch.nn.Softmax(dim=0)
# r0 = sm0(a)
# print(r0.size())
# print(r0)

sm1 = torch.nn.Softmax(dim=1)
r1 = sm1(a)
print(r1.size())
print(r1)
print('Softmax repeat ',sm1(r1))

m1 = torch.max(a, dim=1, keepdim=True)
print('max', m1.values.size())
print(m1)

# sm2 = torch.nn.Softmax(dim=2)
# r2 = sm2(a)
# print(r2.size())
# print(r2)

nllloss = torch.nn.NLLLoss()
celoss = torch.nn.CrossEntropyLoss()
target = torch.tensor([0, 1, 2])
# target = torch.tensor([[0, 1, 1],[0, 1, 1],[0, 1, 1],]) one_hot型报错
l = nllloss(a, target)
cel = celoss(a, target)
print(l)
print(cel)
# cross_entropy等价于Softmax(结果)，再取log，再nllloss
print(nllloss(torch.log(sm1(a)), target))
# pytorch源码中，cross_entropy的实现就是nll_loss(log_softmax(input, dim=1), target……)
# 其中log_softmax是Applies a softmax followed by a logarithm. mathematically equivalent to log(softmax(x))
