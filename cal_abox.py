# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:24:58 2021

@author: Eddie
"""

import glob
import numpy as np
from sklearn.cluster import KMeans
import tqdm

def area(box):
    return box[:, 0] * box[:, 1]

if __name__ == "__main__":
    n_abox = 9
    image_size = 416
    anno_paths = glob.glob("../dataset/VisDrone2019-DET-train/annotations/*.txt")
    
    width = np.empty([0])
    height = np.empty([0])
    print("Loading bbox ...")
    for anno_path in tqdm.tqdm(anno_paths):
        anno_data = np.genfromtxt(anno_path, delimiter=',')
        if len(anno_data.shape) >= 2:
            width = np.concatenate((width, anno_data[:, 2]), axis = 0)
            height = np.concatenate((height, anno_data[:, 3]), axis = 0)
        if len(anno_data.shape) == 1:
            width = np.concatenate((width, anno_data[2:3]), axis = 0)
            height = np.concatenate((height, anno_data[3:4]), axis = 0)
    
    x = np.array([width, height]).transpose()
    print("Kmeans processing ... ({} anchor boxes)".format(n_abox))
    kmeans = KMeans(n_clusters = n_abox)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)
    center = kmeans.cluster_centers_
    
    sort_idx = np.argsort(area(center))
    center = center[sort_idx]
    print(center)
    center += 0.5
    center = center.astype(np.int32)
    print(center)
    np.savetxt('./anchor box.txt', center, delimiter=',', fmt = "%d")
    
    
    
