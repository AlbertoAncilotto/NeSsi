import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import nessi

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 46, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(46)
        self.conv2 = nn.Conv2d(46, 46, 3, groups=46, bias=False)
        self.bn2 = nn.BatchNorm2d(46)
        self.conv3 = nn.Conv2d(46, 16, 1, bias=False)
    
        self.conv4 = nn.Conv2d(16, 46, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(46)
        self.conv5 = nn.Conv2d(46, 46, 3, groups=46, bias=False)
        self.bn4 = nn.BatchNorm2d(46)
        self.conv6 = nn.Conv2d(46, 12, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)
        x = self.conv6(x)
        return x

net = Net()
nessi.get_model_size(net, 'torch' ,input_size=(1,3,320,320))
