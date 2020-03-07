import foolbox
import numpy as np
import torch
import torch.nn.functional as F
from base import affine_params
from base import img_set
from base import model
from base import model_name
from base import target_id

bsize = 20
images, labels = foolbox.utils.samples(dataset=img_set, batchsize=bsize, data_format='channels_first',
                                       bounds=(0, 1))
print('ground truth')
print(labels)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

# 注意文件名后norm的
filename = 'result/aa_{}_{}_targeted{}norm.npy'.format(model_name, img_set, target_id)
data = np.load(filename)

data = torch.from_numpy(data).to(device).float()

output = model.forward(data.to(device))
np_output = output.cpu().detach().numpy()
output_labels = np_output.argmax(axis=-1)
print('output origin ', np.mean(output_labels == labels))
print(output_labels)

results = []
for i in range(len(affine_params)):
    affine_param = torch.from_numpy(affine_params[i]).to(device).float()
    grid = F.affine_grid(affine_param.repeat((data.size()[0], 1, 1)), data.size())
    trans_data = F.grid_sample(data, grid)
    output = model.forward(trans_data.to(device))
    output = F.softmax(output, dim=1)
    np_output = output.cpu().detach().numpy()
    # np.save('result/aa_affine_%d.npy' % i, np_output)
    results.append(np_output)
    print('output {} saved at'.format(i), np.mean(np_output.argmax(axis=-1) == labels))
    print(output.argmax(axis=-1))

results = np.array(results)
np.save('result/aa_affine_results_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id), results)
