import numpy as np
import torch
from base import img_set
from base import model
from base import model_name
from base import target_id

import foolbox

bsize = 20
images, labels = foolbox.utils.samples(dataset=img_set, batchsize=bsize, data_format='channels_first',
                                       bounds=(0, 1))
print('ground truth')
print(labels)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

filename = 'result/aa_{}_{}_targeted{}norm.npy'.format(model_name, img_set, target_id)
data = np.load(filename)
data = torch.from_numpy(data).to(device).float()

# print(data.size())
# print('aa predict label ', model.forward(data).argmax(axis=-1))
output = model.forward(data.to(device))
np_output = output.cpu().detach().numpy()
output_labels = np_output.argmax(axis=-1)
print('output origin ', np.mean(output_labels == labels))
print(output_labels)
