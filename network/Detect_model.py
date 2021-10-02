# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:02:45 2021

@author: Eddie
"""

import torch.nn as nn
from network.Resnet import resnet50

class Detector(nn.Module):
    def __init__(self, num_classes, n_anchors):
        super(Detector, self).__init__()
        self.backbone = resnet50()
        self.out_cha = nn.ModuleList([nn.Conv2d(512, (num_classes + 4 + 1) * n_anchors, 1),
                                      nn.Conv2d(1024, (num_classes + 4 + 1) * n_anchors, 1),
                                      nn.Conv2d(2048, (num_classes + 4 + 1) * n_anchors, 1)])
        
    def forward(self, x):
        features = self.backbone(x)
        output = []
        for layer_id, feature in enumerate(features):
            #print(self.out_cha[layer_id](feature).shape)
            output.append(self.out_cha[layer_id](feature))
        return output
    
if __name__ == "__main__":
    import torch
    model = Detector(20, 3)
    a = torch.randn(1, 3, 416, 416)
    output = model(a)
    