# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:02:45 2021

@author: Eddie
"""

import torch.nn as nn
from network.Resnet import resnet50

class Detector(nn.Module):
    def __init__(self, num_classes, n_abox):
        super(Detector, self).__init__()
        self.backbone = resnet50()
        self.out_cha = nn.Conv2d(2048, (num_classes + 4 + 1) * n_abox, 1)
        
    def forward(self, x):
        feature = self.backbone(x)
        output = self.out_cha(feature)
        return output
    
if __name__ == "__main__":
    import torch
    model = Detector(20, 3)
    a = torch.randn(1, 3, 416, 416)
    output = model(a)
    print(output.shape)
    