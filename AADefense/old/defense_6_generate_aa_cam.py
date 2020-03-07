import json

import cv2
import foolbox
import numpy as np
import torch
from PIL import Image
from base import finalconv_name
from base import img_set
from base import model
from base import model_name
from base import target_id
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

# 正常样本
images, labels = foolbox.utils.samples(dataset=img_set, batchsize=20, data_format='channels_first', bounds=(0, 1))
# 对抗样本，为了获取ground truth的label，上一句也保留
filename = 'result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id)
images = np.load(filename)

# input image
LABELS_PATH = 'labels.json'
IMG_URL = 'http://n.sinaimg.cn/mil/transform/200/w600h400/20190910/f0b4-iekuaqt2521336.jpg'
# 使用本地的图片和下载到本地的labels.json文件
# LABELS_PATH = "labels.json"

net = model
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


# def hook_layer1(module, input, output):
#     layer1.append(output.data.cpu().numpy())
#
#
# def hook_layer2(module, input, output):
#     layer2.append(output.data.cpu().numpy())
#
#
# def hook_layer3(module, input, output):
#     layer3.append(output.data.cpu().numpy())


# layer1 = []
# layer2 = []
# layer3 = []
# net._modules.get('layer1').register_forward_hook(hook_layer1)
# net._modules.get('layer2').register_forward_hook(hook_layer2)
# net._modules.get('layer3').register_forward_hook(hook_layer3)
net._modules.get(finalconv_name).register_forward_hook(hook_feature)
print(features_blobs)
# 得到softmax weight,
params = list(net.parameters())  # 将参数变换为列表
weight_softmax = np.squeeze(params[-2].data.numpy())  # 提取softmax 层的参数


# print(weight_softmax)

# 生成CAM图的函数，完成权重和feature相乘操作
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    # print(type(feature_conv))
    # print(feature_conv.shape) # (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape  # 获取feature_conv特征的尺寸
    output_cam = []
    # class_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
    print(class_idx)
    for idx in class_idx:
        print(np.mean(feature_conv), np.max(feature_conv))
        # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)
        # for m in range(20):
        #     print('W max ', np.max(weight_softmax[m]), ' mean ', np.mean(weight_softmax[idx]))

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))

        # 将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
        print('cam max ', np.max(cam), ' mean ', np.mean(cam))
        # 归一化操作（最小的值为0，最大的为1）
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        # resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


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

for k in range(len(images)):
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
    print(torch.mean(img_variable))
    logit = net(img_variable)
    # 由于前面把层值hook到了features_blobs，此时这个变量里开始有值
    # print(features_blobs)
    # print('layer1', np.mean(layer1))
    # print('layer2', np.mean(layer2))
    # print('layer3', np.mean(layer3))
    # print(np.array(features_blobs).shape) # (1, 1, 512, 7, 7)
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
        print('{:.3f} -> {} {}'.format(probs[i], idx[i], classes[idx[i]]), labels[k])
    # generate class activation mapping for the top1 prediction
    # 输出与图片尺寸一致的CAM图片
    CAMs = returnCAM(features_blobs[k], weight_softmax, [idx[0]])
    # CAMs = returnCAM(features_blobs[0], weight_softmax, [472])
    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    # 将图片和CAM拼接在一起展示定位结果结果
    # img = cv2.imread(img_path)
    # img = img * 255
    height, width, _ = img.shape
    # print(np.max(img))
    # 生成热度图

    # print('cam norm max ', np.max(CAMs[0]/np.sum(CAMs[0])))
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    # print(np.max(heatmap))
    result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite("result/CAM_{}_{}.jpg".format(model_name, k), result)
    cv2.imwrite("result/CAM_aa_t{}_{}_{}.jpg".format(target_id, model_name, k), result)
