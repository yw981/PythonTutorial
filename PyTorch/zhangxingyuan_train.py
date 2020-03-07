#coding=utf-8
#代码参考：https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/finetuning_torchvision_models_tutorial.md
#上述代码讲解：https://www.cnblogs.com/king-lps/p/8665344.html
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.optim import lr_scheduler

##########################################
print("Python Version:",torch.__version__)
print("Torchvision Version:", torchvision.__version__)
##########################################

#参数设置
data_dir = "../tu_berlin/png"
dataset_name = "tu_berlin"
model_name = "vgg"
num_classes = 250
batch_size = 64
num_epochs = 100
feature_extract = False
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" #单块GPU设置
device_ids = [0,1]                        #多块GPU设置
if_use_gpu = 1
###########################################

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-' * 10)

        #每次epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step() #训练的时候进行学习率规划，其定义在下面给出
                model.train()
                print(model)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if if_use_gpu:
                    inputs = inputs.cuda(device=device_ids[0])
                    labels = labels.cuda(device=device_ids[0])

                #将梯度初始化为零
                optimizer.zero_grad()

                #前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    #计算模型输出和损失
                    if is_inception and phase == 'train':
                        #参考链接：https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1+loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    #反向传播+优化, 只在训练阶段
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:4f}'.format(phase, epoch_loss, epoch_acc))

            #深拷贝模型,当验证时遇到了更好的模型则予以保留
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #加载最好的模型
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
#######################################################

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requre_grad = False
########################################################

#重塑网络的模型体系结构
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features #最后fc的输入
        model_ft.fc = nn.Linear(num_ftrs, num_classes) #num_classes是自己数据的类别
        input_size = 224
    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "vgg":
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "inception":
        #注意：此模型图像输入是（299,299）,并有辅助输出
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        #辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        #原始网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
######################################################

#初始化模型进行训练
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)
######################################################

#数据扩充和标准化（训练阶段）
#只进行标准化（验证阶段）
data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

#构建训练集和验证集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train','val']}
#构建训练和验证dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

#将模型放到GPU上
if if_use_gpu:
    model_ft = torch.nn.DataParallel(model_ft, device_ids=device_ids)
    model_ft = model_ft.cuda(device=device_ids[0])

#如果微调,更新所有参数
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.name_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

#观察所有要优化的参数
#https://www.cnblogs.com/king-lps/p/8665344.html
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)

#学习率每7个epoch衰减0.1倍
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80,gamma=0.1)


#设置损失
criterion = nn.CrossEntropyLoss()

#训练和验证
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#保存模型到checkpoint.pth.tar
torch.save(model_ft.state_dict(),"my_"+dataset_name+"_"+model_name+".pth.tar")