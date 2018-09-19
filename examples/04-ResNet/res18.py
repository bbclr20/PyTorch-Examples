import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

from tensorboard_logger import Logger

import matplotlib.pyplot as plt
import numpy as np
from time import time


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.pause(5)


def load_data(dset="MNIST", batch_size=10):
    transform = {
        "CIFAR10": transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "MNIST": transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    }

    classes = {
        "CIFAR10": ("plane", "car", "bird", "cat",
                    "deer", "dog", "frog", "horse", "ship", "truck"),
        "MNIST": ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    }

    if dset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=True,
                                                download=True, transform=transform[dset])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='../../data/CIFAR10', train=False,
                                               download=True, transform=transform[dset])
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
    elif dset == "MNIST":
        trainset = torchvision.datasets.MNIST(root='../../data/MNIST', train=True,
                                              download=True, transform=transform[dset])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root='../../data/MNIST', train=False,
                                             download=True, transform=transform[dset])
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
    return trainloader, testloader, classes[dset]


def visualize_model(model, testloader, classes, num_images=6):
    was_training = model.training  # save the state
    model.eval()  # Sets the module in evaluation mode
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Predicted: {}, Ans:{}'.format(classes[preds[j]],
                                                            classes[labels.tolist()[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)  # restore the state
                    return
        model.train(mode=was_training)   # restore the state


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = torchvision.models.resnet18(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)
    net = net.to(device)

    # trainloader, testloader, classes = load_data("MNIST")
    trainloader, testloader, classes = load_data("CIFAR10")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = 2
    logger = Logger(logdir="./log", flush_secs=2)

    since = time()
    steps = 0
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
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
                steps += i
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                logger.log_value("loss", running_loss / 2000, step=steps)
                running_loss = 0.0

    time_elapsed = time() - since
    input('Training complete in {:.0f}m {:.0f}s. Press any key to continue...'.format(
        time_elapsed // 60, time_elapsed % 60))

    visualize_model(net, testloader, classes)
