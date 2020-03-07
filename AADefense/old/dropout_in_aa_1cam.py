import json

import numpy as np
import torch
from PIL import Image
from base import img_set
from base import model
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

import foolbox

# 正常样本
images, labels = foolbox.utils.samples(dataset=img_set, batchsize=20, data_format='channels_first', bounds=(0, 1))
# 对抗样本，为了获取ground truth的label，上一句也保留
# filename = 'result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id)
# images = np.load(filename)
# print(images.shape)

# 使用本地的图片和下载到本地的labels.json文件
LABELS_PATH = "labels.json"

net = model
features_blobs = []

xx = 8
torch.manual_seed(1234)

D_T = 0.1

# summary(model, (3, 224, 224))
# parm = {}
# for name, parameters in net.named_parameters():
#     print(name, ':', parameters.size())
#     parm[name] = parameters.detach().numpy()


def hook_feature(module, input, output):
    # 直接在这里编辑output的内容会影响其向后传递
    # output *= 9
    # output += 2
    # output += torch.randn(output.size()) / xx
    np_out = output.data.cpu().numpy()
    # print('shape ', np_out.shape)
    # print('finalconv 1  ', np_out.shape, np.mean(np_out), np.var(np_out), np.max(np_out), np.min(np_out))
    min_value = np.where(np_out < D_T, np_out, 0)
    np_out -= min_value
    # print('finalconv 2  ', np_out.shape, np.mean(np_out), np.var(np_out), np.max(np_out), np.min(np_out))
    features_blobs.append(output.data.cpu().numpy())


def hook_layer1(module, input, output):
    # output += torch.randn(output.size()) / xx
    np_out = output.data.cpu().numpy()
    min_value = np.where(np_out < D_T, np_out, 0)
    np_out -= min_value
    layer1.append(output.data.cpu().numpy())


def hook_layer2(module, input, output):
    np_out = output.data.cpu().numpy()
    min_value = np.where(np_out < D_T, np_out, 0)
    np_out -= min_value
    layer2.append(output.data.cpu().numpy())


def hook_layer3(module, input, output):
    np_out = output.data.cpu().numpy()
    min_value = np.where(np_out < D_T, np_out, 0)
    np_out -= min_value
    layer3.append(output.data.cpu().numpy())


# def hook_fc(module, input, output):
#     fc.append(output.data.cpu().numpy())

def hook_avgpool(module, input, output):
    # output = torch.where(output < 0.8, torch.zeros(output.size()),output)
    # output += torch.randn(output.size())/100
    # output = torch.zeros(output.size())
    np_out = output.data.cpu().numpy()
    # print('avgpool  ', np_out.shape, np.mean(np_out), np.var(np_out), np.max(np_out), np.min(np_out))
    # np_out = np.where(np_out < 0.2, 0, np_out)
    # print('avgpool  ', np_out.shape, np.mean(np_out), np.var(np_out), np.max(np_out), np.min(np_out))
    # np_out += 2
    avgpool.append(output.data.cpu().numpy())


layer1 = []
layer2 = []
layer3 = []
# fc = []
avgpool = []
# net._modules.get('layer1').register_forward_hook(hook_layer1)
# net._modules.get('layer2').register_forward_hook(hook_layer2)
# net._modules.get('layer3').register_forward_hook(hook_layer3)
# net._modules.get('fc').register_forward_hook(hook_fc)
# net._modules.get('avgpool').register_forward_hook(hook_avgpool)
# net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# print('fb ',features_blobs)
# 得到softmax weight,
params = list(net.parameters())  # 将参数变换为列表
# print(type(net.parameters()))  # <class 'generator'>
# print(len(params))  # 62
# print(params[-5].data.numpy().shape)  # (512, 512, 3, 3)
# print(params[-4].data.numpy().shape)  # (512,)
# print(params[-3].data.numpy().shape)  # (512,)
# print(params[-2].data.numpy().shape)  # (1000, 512)
# print(params[-1].data.numpy().shape)  # (1000,)
weight_softmax = np.squeeze(params[-2].data.numpy())  # 提取softmax 层的参数
# print(weight_softmax.shape)  # (1000, 512)

# 数据处理，先缩放尺寸到（224*224），再变换数据类型为tensor,最后normalize
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# for k in range(len(images)):
for k in range(20):
    # for k in range(1):
    print(k)
    img = images[k].transpose(1, 2, 0) * 255
    # print(np.mean(img))
    img_pil = Image.fromarray(img.astype('uint8'))
    img_tensor = preprocess(img_pil)
    # print(torch.max(img_tensor))
    # 处理图片为Variable数据
    img_variable = Variable(img_tensor.unsqueeze(0))
    # 将图片输入网络得到预测类别分值
    # print(features_blobs) # 此时值仍为空
    # print('image mean ', torch.mean(img_variable))
    logit = net(img_variable)
    # print('logit shape ', logit.shape)  # torch.Size([1, 1000])
    # 由于前面把层值hook到了features_blobs，此时这个变量里开始有值
    # print('layer1', np.mean(layer1), np.var(layer1), np.max(layer1), np.min(layer1))
    # print('layer2', np.mean(layer2), np.var(layer2), np.max(layer2), np.min(layer2))
    # print('layer3', np.mean(layer3), np.var(layer3), np.max(layer3), np.min(layer3))
    # print('fc ', fc.shape, np.mean(fc), np.var(fc), np.max(fc), np.min(fc))
    # print('avg shape ', np.array(avgpool).shape)
    # print('fb ', np.mean(features_blobs[0]), np.var(features_blobs[0]), np.max(features_blobs[0]), np.min(features_blobs[0]))
    # print(np.array(features_blobs[k]).shape)  # (k, 1, 512, 7, 7)
    # fe = features_blobs[k]
    # re = net.avgpool(fe)
    # re = re.view(re.size(0), -1)
    # re = net.fc(re)
    # print(re)

    # 使用本地的 LABELS_PATH
    with open(LABELS_PATH) as f:
        data = json.load(f).items()
        classes = {int(key): value for (key, value) in data}
    # 使用softmax打分
    h_x = F.softmax(logit, dim=1).data.squeeze()  # 分类分值
    # 对分类的预测类别分值排序，输出预测值和在列表中的位置
    probs, idx = h_x.sort(0, True)
    # 转换数据类型
    probs = probs.numpy()
    idx = idx.numpy()
    # 输出预测分值排名在前五的五个类别的预测分值和对应类别名称
    for i in range(0, 1):
        print('{:.3f} -> {} {} gt: {:.3f} -> {} {}'.format(probs[i], idx[i], classes[idx[i]], h_x[labels[k]], labels[k],
                                                           classes[labels[k]]))
