#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 13:51:59 2021

@author: mmplab603
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def draw_bbox(img, bboxes):
    plt.figure()
    plt.imshow(img)
    
    for bbox in bboxes:
        x = np.array([bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0], bbox[0]])
        y = np.array([bbox[1], bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1]])
        plt.plot(x * img.size[0], y * img.size[0])
