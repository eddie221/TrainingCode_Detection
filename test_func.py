# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:00:04 2021

@author: Eddie
"""

import numpy as np

a = np.arange(2 * 1 * 3 * 4).reshape([2, 1 * 3, 2, 2])
print("before : ", a)
a = np.transpose(a.reshape(2, 3, 1, 2, 2), (0, 2, 3, 4, 1))
print("after : ", a)