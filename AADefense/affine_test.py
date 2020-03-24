import torch
import torch.nn.functional as F
from lenet import Net
from torchvision import datasets, transforms
import numpy as np
from numpy_dataset import NumpyDataset
import math

# 直接写
# affine_params = np.array([
#     [[1.0, 0., 0.1], [0., 1.0, 0.1]],
#     [[1.0, 0., 0.1], [0., 1.0, -0.1]],
#     [[1.0, 0., -0.1], [0., 1.0, 0.1]],
#     [[1.0, 0., -0.1], [0., 1.0, -0.1]],
#     [[1.04, 0., 0.], [0., 1.04, 0.]],
#     [[1.04, 0., 0.], [0., 0.96, 0.]],
#     [[0.96, 0., 0.], [0., 1.04, 0.]],
#     [[0.96, 0., 0.], [0., 0.96, 0.]],
#     [[1.0, 0.04, 0.], [0., 1.0, 0.]],
#     [[1.0, 0., 0.], [0.04, 1.0, 0.]],
# ])


# 按区间生成
start_param = [[1.0, 0., 0], [0., 1.0, 0]]
# stop_param = [[1.0, 0., 0.2], [0., 1.0, 0.2]]
# stop_param = [[1.0, 0., -0.4], [0., 1.0, -0.4]]
stop_param = [[1.4, 0., 1], [0., 1.4, 1]]
# stop_param = [[0.1, 0., 1], [0., 0.1, 1]]
# theta = math.pi / 9  # 旋转theta弧度
# stop_param = [[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0]]
affine_params = np.linspace(start_param, stop_param, num=20)


def restore_model(address):
    # 注意参数和cuda()
    model = Net(1).cuda()

    # model = torch.load(address)
    model.load_state_dict(torch.load(address))
    # model.eval()
    return model


# 自制numpy的数据集


def test(model, criterion, device, test_loader, tag='Test'):
    model.eval()
    # test_loss = 0
    # correct = 0
    results = np.zeros((len(affine_params), 2))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for i in range(len(affine_params)):
                affine_param = torch.from_numpy(affine_params[i]).to(device).float()
                grid = F.affine_grid(affine_param.repeat((data.size()[0], 1, 1)), data.size())
                trans_data = F.grid_sample(data, grid, align_corners=False)

                # 加均一噪声
                # trans_data = trans_data + torch.rand(trans_data.shape).to(device) / 1

                output = model(trans_data)
                batch_loss = criterion(output, target).item()
                results[i][0] += batch_loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                results[i][1] += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    for i in range(len(affine_params)):
        print('{} set {} : Accumulate loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            tag, i, results[i][0], results[i][1], len(test_loader.dataset),
            100. * results[i][1] / len(test_loader.dataset)))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # lenet mnist
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    batch_size = 200
    model_path = '../../model/lenet_mnist.pth'
    model = restore_model(model_path)
    file_path_prefix = 'result/aa_lenet_mnist_fsgm'
    # file_path_prefix = 'result/aa_lenet_mnist_cw'
    file_path_data = file_path_prefix + '.npy'
    file_path_label = file_path_prefix + '_label.npy'
    # AA数据Test set: Average loss: 0.0061, Accuracy: 0 / 10000(0 %)
    # MNIST数据Test set: Average loss: 0.0003, Accuracy: 9789/10000 (98%)


    # densenet 自训练 cifar10
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # batch_size = 200

    # 自定义 Numpy数据 通常是AA样本
    test_loader = torch.utils.data.DataLoader(
        NumpyDataset(file_path_data, file_path_label, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    # MNIST 数据
    test_loader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    # cifar10 数据
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    # ])
    # test_loader_cifar10 = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
    #     batch_size=batch_size, shuffle=False
    # )

    # test(model,criterion, device, train_loader,'Train')
    test(model, criterion, device, test_loader_mnist, 'normal')
    test(model, criterion, device, test_loader, 'AA'+file_path_prefix)
