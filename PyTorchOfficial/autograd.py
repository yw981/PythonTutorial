import torch

x = torch.ones(2, 2)
print(x)

y = x + 2
y.requires_grad_(True)
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

out.backward()

print(x.grad)

print(y.grad)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# print(device)
