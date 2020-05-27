import numpy as np
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from torch import nn
import torch.nn.functional as F


class S2ConvNet_original(nn.Module):

    def __init__(self, PMT_grid):
        super(S2ConvNet_original, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 2

        b_in = 15
        b_l1 = 10
        b_l2 = 6

        grid_s2 = PMT_grid
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1029,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = so3_integrate(x)

        x = self.out_layer(x)

        return x


class S2ConvNet_deep(nn.Module):

    def __init__(self, PMT_grid):
        super(S2ConvNet_deep, self).__init__()

        grid_s2    =  PMT_grid
        grid_so3_1 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_2 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 8, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_3 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 4, n_beta=1, max_gamma=2*np.pi, n_gamma=6)
        grid_so3_4 = so3_near_identity_grid(n_alpha=6, max_beta=np.pi/ 2, n_beta=1, max_gamma=2*np.pi, n_gamma=6)

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in  = 1029,
                nfeature_out = 20,
                b_in  = 15,
                b_out = 15,
                grid=grid_s2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 20,
                nfeature_out = 16,
                b_in  = 15,
                b_out = 8,
                grid=grid_so3_1),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 16,
                b_in  = 8,
                b_out = 8,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 16,
                nfeature_out = 24,
                b_in  = 8,
                b_out = 4,
                grid=grid_so3_2),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 24,
                b_in  = 4,
                b_out = 4,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in  = 24,
                nfeature_out = 32,
                b_in  = 4,
                b_out = 2,
                grid=grid_so3_3),
            nn.ReLU(inplace=False),

            SO3Convolution(
                nfeature_in  = 32,
                nfeature_out = 64,
                b_in  = 2,
                b_out = 2,
                grid=grid_so3_4),
            nn.ReLU(inplace=False)
            )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64,out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x

