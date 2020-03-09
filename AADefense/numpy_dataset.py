import torch.utils.data as data
import numpy as np
from PIL import Image


class NumpyDataset(data.Dataset):
    def __init__(self, data_file_path, label_file_path, transform=None):
        self.transform = transform

        self.test_data = np.load(data_file_path)
        # print('self.test_data.shape', self.test_data.shape)
        self.test_labels = np.load(label_file_path)

    def __getitem__(self, index):
        img, target = self.test_data[index], self.test_labels[index]
        # 此处应该是(H x W x C)，因此调整
        img = img.transpose(1, 2, 0)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')
        # print('img.shape', img.shape)
        # img = Image.fromarray(img, mode='L') # ValueError: Too many dimensions: 3 > 2.报错原因未知

        # 注意transform.ToTensor()操作Converts a PIL Image or numpy.ndarray(H x W x C)
        # to a torch.FloatTensor of shape (C x H x W)
        if self.transform is not None:
            img = self.transform(img)

        # print(img.size())
        return img, target

    def __len__(self):
        return self.test_data.shape[0]
