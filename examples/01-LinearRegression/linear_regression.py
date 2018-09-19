import torch
from torch import nn
import torch.utils.data

from time import time

import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    """ Brutal fitting
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc1(x)
        return out


def create_data(pts=100, fluctuation=1, xrange=10):
    rng = np.random.RandomState(1)
    x = xrange * rng.rand(pts)
    x = np.array([x])
    y = 2 * x - 5 + fluctuation * rng.randn(pts)
    return x.T, y.T


if __name__ == "__main__":
    x, y = create_data()

    if torch.cuda.is_available():
        x_t = torch.tensor(x).type(torch.float).cuda()
        y_t = torch.tensor(y).type(torch.float).cuda()
        net = Net().cuda()
    else:
        x_t = torch.tensor(x).type(torch.float)
        y_t = torch.tensor(y).type(torch.float)
        net = Net()

    optim = torch.optim.Adam(net.parameters(), lr=0.1)

    mse = nn.MSELoss()

    steps = 150
    error = []

    start = time()
    for _ in range(steps):
        optim.zero_grad()

        out = net(x_t)
        loss = mse(out, y_t)
        error.append(loss)
        loss.backward()
        optim.step()
    end = time()
    print("time consumed: ", end-start)

    plt.figure("MSE")
    plt.plot(error)

    plt.figure("Result")
    plt.scatter(x, y)
    x_min, x_max = np.min(x), np.max(x)
    w = net.fc1.cpu().weight.detach().numpy()[0][0]
    b = net.fc1.cpu().bias.detach().numpy()[0]

    print("w(2): {}, b(-5): {}".format(w, b))
    plt.plot([x_min, x_max], [w * x_min + b, w * x_max + b], "r")

    plt.show()

