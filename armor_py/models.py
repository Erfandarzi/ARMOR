from torch import nn


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(4096, 512)
        self.fc_2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 4, kernel_size=5, stride=1)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(4, 10, kernel_size=5, stride=1)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(160, 100)
        self.fc_2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x