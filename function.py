#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 13:51:59 2021

@author: mmplab603
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

def draw_bbox(img, bboxes):
    plt.figure()
    plt.imshow(img)
    
    for bbox in bboxes:
        x = np.array([bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0], bbox[0]])
        y = np.array([bbox[1], bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1]])
        plt.plot(x * img.shape[1], y * img.shape[0], 'r')
        
def bboxes_iou(bboxes_a, bboxes_b, GIoU=False, DIoU=False, CIoU=False):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:], bboxes_b[:, :2] + bboxes_b[:, 2:])
    
    con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
    con_br = torch.max(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:], bboxes_b[:, :2] + bboxes_b[:, 2:])
    
    center_dist = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2] / 2) - (bboxes_b[:, 0] + bboxes_b[:, 2] / 2)) ** 2 / 4 + \
                   ((bboxes_a[:, None, 1] + bboxes_a[:, None, 3] / 2) - (bboxes_b[:, 1] + bboxes_b[:, 3] / 2)) ** 2 / 4
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    
    w1 = bboxes_a[:, 2]
    h1 = bboxes_a[:, 3]
    w2 = bboxes_b[:, 2]
    h2 = bboxes_b[:, 3]
    
    inter_mask = (tl < br).type(tl.type()).prod(dim = 2)
    area_i = torch.prod(br - tl, 2) * inter_mask
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u
    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - center_dist / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (center_dist / c2 + v * alpha)  # CIoU
    return iou