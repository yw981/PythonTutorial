from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch

img_path = "C:/PROJECT/Python/test/c1.png"
img_torch = transforms.ToTensor()(Image.open(img_path))
print(img_torch.shape)

theta = torch.tensor([
    [1, 0, -0.2],
    [0, 1, -0.4]
], dtype=torch.float)
print(theta.shape)
uthe = theta.unsqueeze(0)
print(uthe)
print(uthe.shape)
grid = F.affine_grid(uthe, img_torch.unsqueeze(0).size())
print(grid.shape)
print(grid)
output = F.grid_sample(img_torch.unsqueeze(0), grid)
new_img_torch = output[0]
plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
plt.show()
