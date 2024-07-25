#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
class CNNzijian(nn.Module):
    def __init__(self):
        super(CNNzijian, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1_drop = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2_drop = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(8, 16, 4, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv3_drop = nn.Dropout(0.1)

        self.conv4 = nn.Conv2d(16, 16, 5, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv4_drop = nn.Dropout(0.1)

        self.fc0 = nn.Linear(256, 128)
        self.fc0_drop = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128, 50)
        self.fc1_drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(10, 1)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.conv4_drop(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc0(x))
        x = self.fc0_drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x

