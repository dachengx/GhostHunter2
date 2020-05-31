# -*- coding: utf-8 -*-

import torch
torch.random.manual_seed(1)
from torch import nn

class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        # input is 1029*30
        self.conv1 = nn.Conv2d(1, 30, kernel_size=(9, 1), stride=(3, 1))
        self.conv2 = nn.Conv2d(30, 25, kernel_size=(9, 6), stride=(2, 1))
        self.conv3 = nn.Conv2d(25, 20, kernel_size=(7, 6), stride=(2, 1))
        self.conv4 = nn.Conv2d(20, 15, kernel_size=(7, 6), stride=(2, 1))
        self.conv5 = nn.Conv2d(15, 10, kernel_size=(6, 6), stride=(2, 1))
        self.conv6 = nn.Conv2d(10, 5, kernel_size=(5, 6), stride=(2, 1))
        self.conv7 = nn.Conv2d(5, 1, (6, 5))
        self._initialize_weights()

    def forward(self, x):
        #m = nn.Sigmoid()
        m = nn.LeakyReLU(0.1)
        #m = nn.ELU()
        mf = nn.Softmax(dim=1)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = m(self.conv1(x))
        x = m(self.conv2(x))
        x = m(self.conv3(x))
        x = m(self.conv4(x))
        x = m(self.conv5(x))
        x = m(self.conv6(x))
        x = mf(self.conv7(x).squeeze(3).squeeze(1))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
