import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from test import test
from feature_dataset import FeatureDataset


class Net(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(input_channels, 200)
        self.fc1 = nn.Linear(200, 500)
        self.fc2 = nn.Linear(500, 128)
        self.fc3 = nn.Linear(128, output_channels)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data)
        # print(target)
        optimizer.zero_grad()
        output = model(data)

        # output数据类型tensor float32、维度[batch_size,模型输出维度]此处[64,1000]
        # target均是tensor int64、维度[batch_size]此处[64]
        # torch.nn.CrossEntropyLoss()返回值是function，直接调用torch.nn.CrossEntropyLoss(output, target)是错误的！
        # loss的criterion定义已经提到外部
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        # 每代训练或测试完考虑保存checkpoint，便于resume
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
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    # 加momentum导致STN在MNIST上失效？
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save-path', type=str, default="../../model/feature_model.pth",
                        help='Path and filename to save the trained model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 100
    model_name = 'vgg'
    # out_name = 'imagenet'
    # out_name = 'gaussian'
    out_name = 'uniform'
    file_path_data = '../../data/{}_cifar_{}_feature_data.npy'.format(model_name, out_name)
    file_path_label = '../../data/{}_cifar_{}_feature_label.npy'.format(model_name, out_name)

    # 自定义 Feature数据
    train_loader = torch.utils.data.DataLoader(
        FeatureDataset(file_path_data, file_path_label, train=True),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        FeatureDataset(file_path_data, file_path_label, train=False),
        batch_size=batch_size, shuffle=False, **kwargs)

    model = Net(10, 2).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
    # python train_feature_ood_classifier.py --epochs 150
