import torchvision.models as models
import torch
import numpy as np

model_name = 'resnet18'
# model_name = 'vgg19'
# model_name = 'densenet161'
# model_name = 'squeezenet1_1'
img_set = 'imagenet'
target_id = 456
# target_id = 600

model = []
# 选择使用的网络
if model_name == 'resnet18':
    model = models.resnet18()
    model.load_state_dict(torch.load('../model/resnet18-5c106cde.pth'))
    finalconv_name = 'layer4'
elif model_name == 'vgg19':
    # 暂无法获取最后一层？
    model = models.vgg19()
    model.load_state_dict(torch.load('../model/vgg19-dcbb9e9d.pth'))
    finalconv_name = 'classifier'
# elif model_name == 'densenet161':
#     # 攻击不成功
#     model = models.densenet161(pretrained=True)
#     finalconv_name = 'features'
elif model_name == 'squeezenet1_1':
    model = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features'

model.eval()

affine_params = np.array([
    [[1.0, 0., 0.1], [0., 1.0, 0.1]],
    [[1.0, 0., 0.1], [0., 1.0, -0.1]],
    [[1.0, 0., -0.1], [0., 1.0, 0.1]],
    [[1.0, 0., -0.1], [0., 1.0, -0.1]],
    [[1.04, 0., 0.], [0., 1.04, 0.]],
    [[1.04, 0., 0.], [0., 0.96, 0.]],
    [[0.96, 0., 0.], [0., 1.04, 0.]],
    [[0.96, 0., 0.], [0., 0.96, 0.]],
    [[1.0, 0.04, 0.], [0., 1.0, 0.]],
    [[1.0, 0., 0.], [0.04, 1.0, 0.]],

])



if __name__ == '__main__':
    print(affine_params.shape)
