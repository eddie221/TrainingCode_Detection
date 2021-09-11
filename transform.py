#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:49:13 2021

@author: mmplab603
"""
from PIL import Image, ImageOps
import numpy as np

class RandomFlip():
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, image, bbox):
        if self.prob < np.random.rand(1):
            bbox[:, 0] = 1.0 - bbox[:, 0] - bbox[:, 2]
            image = ImageOps.mirror(image)
            
        return image, bbox
    
class Resize():
    def __init__(self, size):
        if not isinstance(size, tuple):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, image, bbox):
        image = image.resize(self.size)
        return image, bbox

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox
    
