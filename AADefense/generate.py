import foolbox
import numpy as np
import torchvision.models as models
import torch
import torchvision
from torchvision import transforms, datasets
from lenet import Net

# 可编辑部分 模型、数据集、攻击方法
# 实验1： 自训练的LeNet模型，数据未归一化，MNIST全数据集
model = Net(1).cuda()
address = '../../model/lenet_mnist.pth'
model.load_state_dict(torch.load(address))
model = model.eval()
save_file_path_prefix = 'result/aa_lenet_mnist_cw'

# preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10
                                     # , preprocessing=preprocessing
                                     )
# attack = foolbox.attacks.FGSM(fmodel)
attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel,
                                               # criterion=foolbox.criteria.TargetClassProbability(target_id, p=.2)
                                               )

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data', train=True, download=True,
#                    transform=transform
#                    # transform=torchvision.transforms.ToTensor()
#                    ),
#     batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False,
                   transform=transform
                   # transform=torchvision.transforms.ToTensor()
                   ),
    batch_size=200, shuffle=False)


# get a batch of images and labels and print the accuracy
# images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
# print(type(images), type(labels))
# print(np.max(images), np.mean(images), np.var(images), np.min(images))
# print(images.shape, labels.shape)
# print(labels)
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# 1.0 0.4643064 0.06226354 0.0
# (16, 3, 224, 224) (16,)
# [243 559 438 990 949 853 609 609 915 455 541 630 741 471 129  99]



# cifar10 数据
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
# ])
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(root='../../data', train=True, download=True,
#                      transform=transform),
#     batch_size=64, shuffle=False
# )
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform),
#     batch_size=64, shuffle=False
# )
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# (64, 3, 32, 32) (64,)
# 1.0 0.45431957 0.24718454 0.0


results = []
labels = []
count_origin_right = 0
count_aa_failed = 0

for batch_idx, (data, target) in enumerate(test_loader):
    np_data = data.numpy()
    np_target = target.numpy()
    origin_result = fmodel.forward(np_data).argmax(axis=-1) == np_target
    count_origin_right += np.count_nonzero(origin_result)
    print('origin acc = ', np.mean(origin_result), ' total right ', count_origin_right)

    # unpack=True只返回numpy array的攻击结果图片，unpack=False否则返回对象，包括标签等所有信息
    adversarials = attack(np_data, np_target, unpack=True)
    # TODO 考虑只攻击分类对的图片，把错误的剔除，并同时保存对应的ground truth标签
    # https://foolbox.readthedocs.io/en/stable/modules/adversarial.html

    test_result = fmodel.forward(adversarials).argmax(axis=-1) == np_target
    count_aa_failed += np.count_nonzero(test_result)

    results.append(adversarials)
    # print(target.shape) # torch.Size([200])
    # TODO 考虑Tensor的合并
    labels.append(target.numpy())
    print('finish ', batch_idx, ' rate = ', np.mean(test_result), ' total failed ', count_aa_failed)
    # if batch_idx == 2:
    #     break

    # 画图查看，需保证结果unpack=False利用Adv对象
    # cnt = 0
    # row = 16
    # col = 8
    # plt.figure(figsize=(col, row))
    # for i in range(row):
    #     for j in range(col):
    #         cnt += 1
    #         plt.subplot(row, col, cnt)
    #         plt.xticks([], [])
    #         plt.yticks([], [])
    #
    #         ex = np.squeeze(adversarials[i * col + j].perturbed)
    #         plt.title("{} -> {}".format(adversarials[i * col + j].original_class,
    #                                     adversarials[i * col + j].adversarial_class))
    #         plt.imshow(ex, cmap="gray")
    # plt.tight_layout()
    # plt.show()

results = np.array(results)
# print(results.shape)
results = results.reshape(-1, 1, 28, 28)
# print(results.shape)

labels = np.array(labels)
labels = labels.flatten()
# print(labels.shape)
np.save(save_file_path_prefix+'.npy', results)
np.save(save_file_path_prefix+'_label.npy', labels)
