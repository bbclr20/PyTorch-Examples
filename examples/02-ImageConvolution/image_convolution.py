from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

to_tensor = ToTensor()
to_pil = ToPILImage("L")

# original image
lena = Image.open("../../data/images/lena.jpg").convert("L")
lena.show()

input = to_tensor(lena).unsqueeze(0)

# blur image
blur = torch.ones(1, 1, 8, 8, dtype=torch.float)/64
conv = nn.Conv2d(1, 1, 3, 1, bias=False)
conv.weight.data = blur
output = conv(input)
to_pil(F.relu(output.data.squeeze(0))).show()

# edge detection
edge = torch.tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]], dtype=torch.float)
conv = nn.Conv2d(1, 1, 3, 1, bias=False)
conv.weight.data = edge
output = conv(input)
to_pil(F.relu(output.data.squeeze(0))).show()
