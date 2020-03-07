import torch
import torch.nn.functional as F
# from lenet import Net
from torchvision import datasets, transforms
# from torchvision.models.densenet import DenseNet as Net
import torchvision


def restore_model():
    # model = Net()
    # model = torchvision.models.resnet18()
    model = torchvision.models.densenet161()

    # 没使用GPU训练模型参数保存conv1.weight 使用了GPU训练模型参数会被保存成module.conv1.weight
    model = torch.nn.DataParallel(model)

    address = './model_best.pth.tar'
    model_dict = torch.load(address)
    # print(model_dict)
    # model.load_state_dict(model_dict)
    model.load_state_dict(model_dict['state_dict'])
    # model.eval()

    return model


def test(model, device, test_loader, tag='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        tag, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 64
    model = restore_model()

    # MNIST
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)

    # cifar10 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )

    # test(model, device, train_loader,'Train')
    test(model, device, test_loader)
