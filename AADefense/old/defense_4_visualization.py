import foolbox
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from base import img_set
from base import model_name
from base import target_id


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    # print(inp.size())
    inp = inp.numpy().transpose((1, 2, 0))
    # print(inp.shape)
    return inp


images, labels = foolbox.utils.samples(dataset=img_set, batchsize=20, data_format='channels_first', bounds=(0, 1))
# filename = 'result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id)
filename = 'result/aa_{}_{}_targeted{}norm.npy'.format(model_name, img_set, target_id)
images = np.load(filename)

images = torch.from_numpy(images).float()
grid_images = convert_image_np(torchvision.utils.make_grid(images, nrow=10))
plt.imshow(grid_images)
plt.show()

# for i in range(9):
#     affine_param = torch.from_numpy(affine_params[i]).float()
#     print(affine_param)
#     # print(type(images))
#     grid = F.affine_grid(affine_param.repeat((images.size()[0], 1, 1)), images.size())
#     trans_images = F.grid_sample(images, grid, padding_mode="reflection")  # reflection border
#     # print(results[i].shape)
#     grid_images = convert_image_np(torchvision.utils.make_grid(trans_images, nrow=10))
#     plt.imshow(grid_images)
#     # plt.set_title('Pred', str(results[i]))
#
#     plt.show()
