import foolbox
import numpy as np
import torch
import torch.nn.functional as F
from base import img_set
from base import model
from base import model_name
from base import target_id

start_param = [[1.0, 0., 0], [0., 1.0, 0]]
# stop_param = [[1.0, 0., 0.4], [0., 1.0, 0.4]]
# stop_param = [[1.0, 0., -0.4], [0., 1.0, -0.4]]
stop_param = [[1.4, 0., 1], [0., 1.4, 1]]
# stop_param = [[0.1, 0., 1], [0., 0.1, 1]]
# stop_param = [[1.0, 1, 0.], [0., 1.0, 0]]
affine_params = np.linspace(start_param, stop_param, num=20)

_, labels = foolbox.utils.samples(dataset=img_set, batchsize=20, data_format='channels_first', bounds=(0, 1))
print('ground truth')
print(labels)

filename = 'result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id)
images = np.load(filename)

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))

torch_images = torch.from_numpy(images).float()

accs = []
for i in range(len(affine_params)):
    affine_param = torch.from_numpy(affine_params[i]).float()
    # print(affine_param)
    # print(type(images))
    grid = F.affine_grid(affine_param.repeat((torch_images.size()[0], 1, 1)), torch_images.size())
    trans_images = F.grid_sample(torch_images, grid)  # , padding_mode="reflection" reflection border
    pred = fmodel.forward(trans_images.detach().numpy())
    # pred = F.softmax(pred, dim=1)
    # np_pred = pred.detach().numpy()
    acc = np.mean(pred.argmax(axis=-1) == labels)
    print('output {} saved at'.format(i), acc)
    accs.append(acc)
    # print(pred.argmax(axis=-1))
    # grid_images = convert_image_np(torchvision.utils.make_grid(trans_images, nrow=10))
    # plt.imshow(grid_images)
    # plt.set_title('Pred', str(results[i]))
    # plt.show()

print(','.join(str(acc) for acc in accs))
