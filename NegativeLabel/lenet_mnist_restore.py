from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.fcout = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fcout(x)
        return torch.sigmoid(x)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output)
            print(torch.min(output,dim=1,keepdim=True))

            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmin(dim=1, keepdim=True)  # get the index of the max log-probability
            # print(output[pred])
            # print('label ',target)

            gaussian = torch.rand(data.size()).to(device)
            gaussian_output = model(gaussian)
            print(gaussian_output)
            print(torch.min(gaussian_output,dim=1,keepdim=True))
            # noise_pred = noise_output.argmin(dim=1, keepdim=True)
            # print(noise_output[noise_pred])
            uniform = torch.rand(data.size()).to(device)
            uniform_output = model(uniform)
            print(uniform_output)
            print(torch.min(uniform_output,dim=1,keepdim=True))
            exit(0)

            # correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    batch_size = 4
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    epochs = 2

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load('../../model/lenet_mnist_nl_model.pth'))
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #
    # for epoch in range(1, epochs + 1):
    #     train(model, device, train_loader, optimizer, epoch)
    #     print(model.apms)
    test(model, device, test_loader)
    # print(model.state_dict())

    # torch.save(model.state_dict(), "../model/lenet_mnist_affine_model.pth")


if __name__ == '__main__':
    main()
