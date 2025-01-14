import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils_func import softmax

def softmax(data):
    for i in range(data.shape[0]):
        f = data[i,:].reshape (data.shape[1])
        data[i,:] = torch.exp(f) / torch.sum(torch.exp(f))
    return data

class tile2openpose_conv3d(nn.Module):
    def __init__(self, windowSize):
        super(tile2openpose_conv3d, self).__init__()   #tactile 64*64
        if windowSize == 0:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(2*windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        # 32*64*64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)) # 48 * 48
        # 64*32*32

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        #128*32*32


        self.l1 = nn.Sequential(
            nn.Linear(128*32*32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.l2 = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

        self.c1 = nn.Sequential(
            nn.Linear(128*32*32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.c2 = nn.Sequential(
            nn.Linear(512, 5),
        )


    def forward(self, input, device):
        output = self.conv_0(input)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = output.view(input.shape[0], 128*32*32)

        sp_ag = self.l1(output)
        sp_ag = self.l2(sp_ag)

        classifi = self.c1(output)
        classifi = self.c2(classifi)



        return sp_ag, classifi


