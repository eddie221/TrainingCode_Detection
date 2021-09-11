#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 23:05:27 2021

@author: mmplab603
"""

from torch.utils.data import Dataset
import glob
import os
from numpy import genfromtxt
from PIL import Image
import numpy as np

# =============================================================================
#  Annotation information
#  <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box
# 
#  <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box
# 
#  <bbox_width>	     The width in pixels of the predicted object bounding box
# 
# <bbox_height>	     The height in pixels of the predicted object bounding box
# 
#    <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
#                      an object instance.
#                      The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
#                      while 0 indicates the bounding box will be ignored.
#                       
# <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
#                      people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
#                      others(11))
#                       
# <truncation>	     The score in the DETECTION result file should be set to the constant -1.
#                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
#                      (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
#                       
# <occlusion>	     The score in the DETECTION file should be set to the constant -1.
#                      The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
#                      (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
#                      (occlusion ratio 50% ~ 100%)).
# =============================================================================

class Detection_dataset(Dataset):
    def __init__(self, image_path, anno_path, transform):
        super(Detection_dataset, self).__init__()
        self.anno_path = anno_path
        self.image_pool = glob.glob(os.path.join(image_path, "*.jpg"))
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_pool[index]
        image = Image.open(image_path)
        image_name = image_path.split('/')[-1][:-4]
        anno_data = genfromtxt(os.path.join(self.anno_path, "{}.txt".format(image_name)), delimiter=',')
        anno_data[:, 1:4:2] /= image.size[1]
        anno_data[:, 0:4:2] /= image.size[0]
        anno_data = anno_data[:-3]
        assert anno_data[:, :4].min() >= 0, "Bounding box smaller than zero. {}".format(image_name)
        assert anno_data[:, :4].max() <= 1, "Bounding box smaller than zero. {}".format(image_name)
        image, anno_data = transform(image, anno_data)
        
        return image, anno_data

if __name__ == '__main__':
    from function import draw_bbox
    from transform import *
    transform = Compose([Resize((1024, 1024)), RandomFlip(0.1)])
    dataset = Detection_dataset("../datasets/visdrone/VisDrone2019-DET-train/images", "../datasets/visdrone/VisDrone2019-DET-train/annotations", transform)
    img, anno = dataset[100]
    draw_bbox(img, anno)