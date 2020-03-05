import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from lenet import Net
# from stn_lenet import Net
from torchvision.models.densenet import DenseNet as Net
from test import test


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        # loss = F.cross_entropy(output, target)

        # output数据类型tensor float32、维度[batch_size,模型输出维度]此处[64,1000]
        # target均是tensor int64、维度[batch_size]此处[64]
        # torch.nn.CrossEntropyLoss()返回值是function，直接调用torch.nn.CrossEntropyLoss(output, target)是错误的！
        # loss的criterion定义已经提到外部
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save-path', type=str, default="../../model/densenet_c10_model.pth",
                        help='Path and filename to save the trained model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    # mnist 数据
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../../data', train=True, download=True,
    #                    transform=transform),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../../data', train=False, transform=transform),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # cifar10 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data', train=True, download=True,
                         transform=transform),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False
    )

    # 各模型新建时，注意参数
    model = Net(3, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(model, criterion, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), args.save_path)
        print('model saved to ', args.save_path)


if __name__ == '__main__':
    main()
