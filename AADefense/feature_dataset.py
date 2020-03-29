import torch.utils.data as data
import numpy as np


class FeatureDataset(data.Dataset):
    def __init__(self, data_file_path, label_file_path, train=True):
        self.train = train

        sep = 8000
        self.data = np.load(data_file_path)
        self.train_data = np.vstack((self.data[0:sep], self.data[10000:10000+sep]))
        # print(self.data.shape)
        self.test_data = np.vstack((self.data[sep:10000], self.data[10000+sep:20000]))
        # print(self.test_data.shape)
        # print('self.test_data.shape', self.test_data.shape)
        self.labels = np.load(label_file_path)
        # print(self.labels)
        # print(self.labels.shape)
        self.train_labals = np.append(self.labels[0:sep], self.labels[10000:10000+sep])
        # self.train_labals = self.labels[0:4000,5000:9000]
        # print(self.labels[0:4000])
        # print(self.labels[5000:9000])
        self.test_labals = np.append(self.labels[sep:10000], self.labels[10000+sep:20000])
        # print(self.test_labals)
        # exit(0)

        if self.train:
            self.data = self.train_data
            self.labels = self.train_labals
        else:
            self.data = self.test_data
            self.labels = self.test_labals

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')
        # print('img.shape', img.shape)
        # img = Image.fromarray(img, mode='L') # ValueError: Too many dimensions: 3 > 2.报错原因未知

        # 注意transform.ToTensor()操作Converts a PIL Image or numpy.ndarray(H x W x C)
        # to a torch.FloatTensor of shape (C x H x W)
        # if self.transform is not None:
        #     img = self.transform(img)

        # print(img.size())
        return img, target

    def __len__(self):
        return self.data.shape[0]
