#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


def loss(output_image, Input_Image):
    # print(f"check data type of output_iamge{output_iamge}")
    loss = torch.mean(torch.pow((Input_Image -  output_image), 2))
    # img2mse = lambda x, y : torch.mean((x - y) ** 2)
    # img_loss = torch.mean((train_image - gt_image)**2)
    return loss



class NeRF(nn.Module):
#     """
    
#     """
#     def __init__(self,input_channels, width):
#         super().__init__()
#         self.layer_1 = nn.Sequential(nn.Linear(input_channels, width, dtype=torch.float64), nn.ReLU())
#         self.layer_2 = nn.Sequential(nn.Linear(width, width), nn.ReLU())
#         self.layer_3 = nn.Sequential(nn.Linear(width + input_channels, width), nn.ReLU())
#         self.layer_4 = nn.Sequential(nn.Linear(width,4))


#     def forward(self, x):
#         x = x.to(torch.float64)
#         res = x
#         x = self.layer_1(x)
#         # x1 = self.layer_2(x)
#         x2= torch.concat([x, res], axis = -1)
#         x3 = self.layer_3(x2)
#         # x = self.layer_2(x)
#         x4 = self.layer_4(x3)
#         print("shapeof x", x4.shape)

#         return x4
    # def __init__(self, input_channels, width):
    #     super().__init__()
    #     self.layer_1 = nn.Sequential(nn.Linear(input_channels, width, dtype=torch.float64), nn.ReLU())
    #     self.layer_2 = nn.Sequential(nn.Linear(width, width, dtype=torch.float64), nn.ReLU())
    #     self.layer_3 = nn.Sequential(nn.Linear(width + input_channels, width, dtype=torch.float64), nn.ReLU())
    #     self.layer_4 = nn.Sequential(nn.Linear(width, 4, dtype=torch.float64))

    # def forward(self, x):
    #     res = x
    #     x = self.layer_1(x)
    #     # x2 = torch.cat([x, res], axis=-1)
    #     x3 = self.layer_2(x)
    #     x4 = self.layer_4(x3)
    #     # print("shape of x", x4.shape)
    #     return x4

    def __init__(self, input_channels, width):
        super().__init__()
        self.layer_1 = nn.Sequential(nn.Linear(input_channels, width, dtype=torch.double), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(width, width, dtype=torch.double), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Linear(width + input_channels, width, dtype=torch.double), nn.ReLU())
        self.layer_4 = nn.Sequential(nn.Linear(width, 4, dtype=torch.double))

    def forward(self, x):
        res = x
        x = self.layer_1(x)
        # x2 = torch.cat([x, res], axis=-1)
        x3 = self.layer_2(x)
        x4 = self.layer_4(x3)
        # print("shape of x", x4.shape)
        return x4