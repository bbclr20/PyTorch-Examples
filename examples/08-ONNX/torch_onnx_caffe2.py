import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from time import time
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_caffe2.backend


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        x = x.view(-1, 16*4*4)
        x = self.classifier(x)
        return x

    def load_data(self, batch_size=100):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
    net = Net().to(device)

    batch_size = 10
    trainloader, testloader, classes = net.load_data(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": net.features.parameters(), "lr": 1e-2},
        {"params": net.classifier.parameters()}], lr=1e-3)

    EPOCHS = 1
    LOSS = []

    start = time()
    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / batch_size))
                LOSS.append(running_loss)
                running_loss = 0.0
    end = time()
    print("time consumption: ", end - start)
    # plt.plot(LOSS)

    #
    # test with images
    #
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    del dataiter

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    plt.show()

    images = images.to(device)
    labels = labels.to(device)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
        for j in range(batch_size)))

    train_result = outputs.cpu().detach().numpy()

    dummy_input = Variable(torch.randn(batch_size, 1, 28, 28)).to(device)

    #
    # Export ONNX model with PyTorch and load the model with Caffe2 as the backend
    #
    torch.onnx.export(net, dummy_input, "lenet.onnx", export_params=True)

    model = onnx.load("lenet.onnx")
    prepared_backend = onnx_caffe2.backend.prepare(model)
    W = {model.graph.input[0].name: images.cpu().detach().numpy()}
    load_out = prepared_backend.run(W)[0]
    np.testing.assert_almost_equal(train_result, load_out, decimal=3)
