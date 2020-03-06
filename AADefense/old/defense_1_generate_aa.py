import time

import foolbox
import numpy as np
from base import img_set
from base import model
from base import model_name
from base import target_id


preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

bsize = 20
# get a batch of images and labels and print the accuracy
images, labels = foolbox.utils.samples(dataset=img_set, batchsize=bsize, data_format='channels_first', bounds=(0, 1))
print('ground thuth label ', labels)
origin_result = fmodel.forward(images)
print('origin predict label ', origin_result.argmax(axis=-1))
print('origin predict acc ', np.mean(origin_result.argmax(axis=-1) == labels))

start = time.process_time()

# apply the attack
attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel,
                                               criterion=foolbox.criteria.TargetClassProbability(target_id, p=.2)
                                               )
adversarials = attack(images, labels)
# if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

elapsed = (time.process_time() - start)
print("Time used:", elapsed)

np.save('result/aa_{}_{}_targeted{}.npy'.format(model_name, img_set, target_id), adversarials)

aa_result = fmodel.forward(adversarials)
# Foolbox guarantees that all returned adversarials are in fact in adversarials
print('attacked label ', aa_result.argmax(axis=-1))
print('attacked acc ', np.mean(aa_result.argmax(axis=-1) == labels))
# -> 0.0
