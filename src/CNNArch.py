import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, params):
        super(CNNModel, self).__init__()
        self.params = params
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(5,5))
        self.relu2 = nn.ReLU(True)


        self.fc1  = nn.Linear(16*16, 64)
        self.relu3 = nn.ReLU(True)
        self.fc2 = nn.Linear(64, 11)
        self.softmax = nn.Softmax()

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(50)
        self.bn3 = nn.BatchNorm1d(100)

        self.dropout = nn.Dropout(p=self.params['dropout'])

    def forward(self, x):

        x = self.conv1(x)
        if self.params['pool1']:
            x = self.pool1(x)
        x = self.relu1(x)
        if self.params['bn']:
            x = self.bn1(x)

        x = self.conv2(x)
        if self.params['pool2']:
            x = self.pool2(x)
        x = self.relu2(x)
        if self.params['bn']:
            x = self.bn2(x)

        batch_size = x.size()[0]
        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.relu3(x)
        if self.params['bn']:
            x = self.bn3(x)

        if self.params['dropout'] > 0:
            x = self.droput(x)

        x = self.fc2(x)
        out = self.softmax(x)
        return out