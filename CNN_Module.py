# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        # input is 1000*30
        self.conv1 = nn.Conv2d(1, 15, (15, 10), stride=(5, 1))
        self.conv2 = nn.Conv2d(15, 10, (10, 9), stride=(2, 1))
        self.conv3 = nn.Conv2d(10, 5, (9, 8))
        self.conv4 = nn.Conv2d(1, 1, (6, 6))

    def forward(self, x):
        m = nn.Sigmoid()
        drop_out = nn.Dropout(0.9)
        x = m(self.conv1(x))
        x = m(self.conv2(x))
        x = m(self.conv3(x))
        x = F.sigmoid(self.conv4(x))
        return x
