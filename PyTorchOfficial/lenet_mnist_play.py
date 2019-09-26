import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    return inp


use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 16

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

data, label = next(iter(test_loader))
print(len(data))
# print(data)
print(label)

in_grid = torchvision.utils.make_grid(data, nrow=4)
print(in_grid.shape)
# np.transpose的理解，transpose(1,2,0)就是把原来的第1轴->0位置，第2轴->1位置，第0轴->2位置，即(3,122,122)->(122,122,3)
res = in_grid.numpy().transpose((1, 2, 0))
print(res.shape)
# plt.imshow(res)
# plt.show()

np.random.seed(1234)
apms = np.random.normal(size=(10, 2, 3))
print(apms[1])

# print(data.shape)

# for batch_idx, (data, target) in enumerate(train_loader):
#     data, target = data.to(device), target.to(device)
#     if batch_idx % 1000 == 0:
#         print(batch_idx)
#         print(data.shape)
#         print(target.shape)
#         in_grid = torchvision.utils.make_grid(data.cpu(),nrow=4)
#         plt.imshow(in_grid.numpy().transpose((1, 2, 0)))
#         plt.show()
#         exit(0)


# in_grid = convert_image_np(torchvision.utils.make_grid(data))
#
# out_grid = convert_image_np(torchvision.utils.make_grid(data))
#
# # Plot the results side-by-side
# f, axarr = plt.subplots(1, 2)
# axarr[0].imshow(in_grid)
# axarr[0].set_title('Dataset Images')
#
# axarr[1].imshow(out_grid)
# axarr[1].set_title('Transformed Images')
