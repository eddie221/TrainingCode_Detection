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
from torchvision.transforms.functional import to_tensor
from transform import *
import torch

# =============================================================================
#  Annotation information --- https://github.com/VisDrone/VisDrone2018-DET-toolkit
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
        image = np.array(Image.open(image_path))
        image_path = image_path.replace("\\", "/")
        image_name = image_path.split('/')[-1][:-4]
        
        anno_data = genfromtxt(os.path.join(self.anno_path, "{}.txt".format(image_name)), delimiter=',')
        if len(anno_data.shape) == 1:
            anno_data = np.expand_dims(anno_data, axis = 0)
        anno_data[:, 1:4:2] /= image.shape[0]
        anno_data[:, 0:4:2] /= image.shape[1]
        assert anno_data[:, :4].min() >= 0, "Bounding box smaller than zero. {}".format(image_name)
        assert anno_data[:, :4].max() <= 1, "Bounding box smaller than zero. {}".format(image_name)
        image, anno_data = self.transform(image, anno_data)
        image_norm = to_tensor(image)
        image_norm = Normalize()(image_norm)
        
        # get annotation data x, y, w, h, category
        anno_data = anno_data[:, [0, 1, 2, 3, 5]]
# =============================================================================
#         anno_data[:, 0] = (anno_data[:, 0] + anno_data[:, 2]) / 2
#         anno_data[:, 1] = (anno_data[:, 1] + anno_data[:, 3]) / 2
# =============================================================================
        return image_norm.type(torch.FloatTensor), torch.from_numpy(anno_data), image
    
    def __len__(self):
        return len(self.image_pool)
    
def collate(batch):
    images_norm = []
    images = []
    annos = []
    for img_norm, anno, img in batch:
        images_norm.append(np.expand_dims(img_norm, axis = 0))
        annos.append(anno)
        images.append([img])
    
    images_norm = np.concatenate(images_norm, axis = 0)
    images_norm = torch.from_numpy(images_norm)
    
    images = np.concatenate(images, axis = 0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images_norm, annos, images

if __name__ == '__main__':
    from function import draw_bbox
    from transform import *
    transform = Compose([Resize((1024, 1024)), RandomFlip(0.1)])
    dataset = Detection_dataset("../dataset/VisDrone2019-DET-train/images", "../dataset/VisDrone2019-DET-train/annotations", transform)
    img_norm, anno, img = dataset[100]
    draw_bbox(img, anno)
    print(img.shape)