import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
from time import time


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self, dset="MNIST"):
        super(Net, self).__init__()
        self.dset = dset
        if self.dset == "CIFAR10":  # input image size [3, 32, 32]
            self.features = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )
        elif self.dset == "MNIST":  # input image size [1, 28, 28]
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )

    def forward(self, x):
        x = self.features(x)
        if self.dset == "CIFAR10":
            x = x.view(-1, 16 * 5 * 5)
        elif self.dset == "MNIST":
            x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x

    def load_data(self, batch_size=10):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if self.dset == "CIFAR10":
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            trainset = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=True,
                                                    download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=False,
                                                   download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
        elif self.dset == "MNIST":
            classes = ('0', '1', '2', '3',
                       '4', '5', '6', '7', '8', '9')

            trainset = torchvision.datasets.MNIST(root='../../data/MNIST', train=True,
                                                  download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=True, num_workers=2)

            testset = torchvision.datasets.MNIST(root='../../data/MNIST', train=False,
                                                 download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                     shuffle=True, num_workers=2)
        return trainloader, testloader, classes


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = Net("MNIST").to(device)
    # net = Net("CIFAR10").to(device)

    trainloader, testloader, classes = net.load_data()

    criterion = nn.CrossEntropyLoss()
# [start-20180905-ben-mod]#
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.Adam([
        {"params": net.features.parameters(), "lr": 1e-2},
        {"params": net.classifier.parameters()}], lr=1e-3)
# [end-20180905-ben-mod]#
    EPOCHS = 2
    LOSS = []

    start = time()
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                LOSS.append(running_loss)
                running_loss = 0.0
    end = time()
    print("time consumption: ", end-start)
    
    plt.plot(LOSS)
    plt.show()

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    del dataiter

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

    images = images.cuda()
    labels = labels.cuda()

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
            for j in range(10)))
    plt.show()
