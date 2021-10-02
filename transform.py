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
            image = image[:, ::-1, :].copy()
            
        return image, bbox
    
class Resize():
    def __init__(self, size):
        if not isinstance(size, tuple):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, image, bbox):
        image = Image.fromarray(image)
        image = image.resize(self.size)
        image = np.array(image)
        return image, bbox
    
class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image):
        image = (image - np.expand_dims(np.expand_dims(self.mean, axis = -1), axis = -1)) / np.expand_dims(np.expand_dims(self.std, axis = -1), axis = -1)
        return image

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox):
        for t in self.transforms:
            image, bbox = t(image, bbox)
        return image, bbox
    
