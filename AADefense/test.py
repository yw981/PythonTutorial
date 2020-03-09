import torch
import torch.nn.functional as F
from lenet import Net
from torchvision import datasets, transforms
# from torchvision.models.densenet import DenseNet as Net
from numpy_dataset import NumpyDataset


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
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print(data.size())
            # print(target.size())
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            batch_loss = criterion(output, target).item()
            test_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        tag, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # lenet mnist
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 200
    model_path = '../../model/lenet_mnist.pth'
    model = restore_model(model_path)
    file_path_prefix = 'result/aa_lenet_mnist_fsgm'
    file_path_data = file_path_prefix + '.npy'
    file_path_label = file_path_prefix + '_label.npy'
    # AA数据Test set: Average loss: 0.0061, Accuracy: 0 / 10000(0 %)
    # MNIST数据Test set: Average loss: 0.0003, Accuracy: 9789/10000 (98%)


    # densenet 自训练 cifar10
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # batch_size = 200

    # 自定义 Numpy数据
    # test_loader = torch.utils.data.DataLoader(
    #     NumpyDataset(file_path_data, file_path_label, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=batch_size, shuffle=False, **kwargs)

    # MNIST 数据
    test_loader = torch.utils.data.DataLoader(
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
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
    #     batch_size=batch_size, shuffle=False
    # )

    # test(model,criterion, device, train_loader,'Train')
    test(model, criterion, device, test_loader)
