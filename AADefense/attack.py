import foolbox
import numpy as np
import torchvision.models as models
import torch
import time
from test import restore_model
import torch.nn.functional as F
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
batch_size = 64

model = restore_model()

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

preprocessing = dict(mean=[0.1307], std=[0.3081])
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

test_loader = foolbox.utils.samples(dataset='mnist', batchsize=batch_size, data_format='channels_first', bounds=(0, 1))

test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    print(type(data))
    output = fmodel.forward(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\n set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# # get a batch of images and labels and print the accuracy
# images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=20, data_format='channels_first', bounds=(0, 1))
# print('label ', labels)
# print(fmodel.forward(images).argmax(axis=-1))
# -> 0.9375


# start = time.process_time()
#
# # apply the attack
# # attack = foolbox.attacks.FGSM(fmodel)
# # adversarials = attack(images, labels)
# attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel,
#                                                # criterion=foolbox.criteria.TargetClassProbability(781, p=.5)
#                                                )
# adversarials = attack(images, labels)
# # if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# # if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan
#
# elapsed = (time.process_time() - start)
# print("Time used:", elapsed)
#
# np.save('result/adversarials.npy', adversarials)
#
# # Foolbox guarantees that all returned adversarials are in fact in adversarials
# print(fmodel.forward(adversarials).argmax(axis=-1))
# # print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
# # -> 0.0
