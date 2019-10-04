import torch.nn as nn
import torch.nn.functional as F


class MNistFlat(nn.Module):
    def __init__(self, p_drop=0.1, n_hidden_1=512, n_hidden_2=128):
        super(MNistFlat, self).__init__()
        self.fc1 = nn.Linear(28*28, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28*28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class MNistNet(nn.Module):
    def __init__(self, p_drop=0.1, n_filter_1=16, n_filter_2=32, n_hidden=256):
        super(MNistNet, self).__init__()
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filter_1, kernel_size=5, stride=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=n_filter_1, out_channels=n_filter_2, kernel_size=5, stride=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2)
        self.drop2d = nn.Dropout2d(p=p_drop)
        self.fc1 = nn.Linear(4*4*self.n_filter_2, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxp1(x))
        x = self.conv2(x)
        x = self.drop2d(x)
        x = F.relu(self.maxp2(x))

        x = x.view(-1, 4*4*self.n_filter_2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
