from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, ToTensor, \
    Resize, RandomAffine, ColorJitter
# from dataset import MaskDataset, get_img_files
import numpy as np
import matplotlib.pyplot as plt
import torchvision

img_size = 240
BATCH_SIZE = 32
train_files = "../../data/hymenoptera_data/train"

train_transform = Compose([
        ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomResizedCrop(img_size, scale=(0.8, 1.2)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

image_datasets = {"train": ImageFolder(train_files, train_transform)}
train_loader = DataLoader(image_datasets["train"],
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=4)

images, labels = iter(train_loader).__next__()
img = torchvision.utils.make_grid(images)
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()
