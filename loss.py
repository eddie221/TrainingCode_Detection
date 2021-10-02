# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:24:12 2021

@author: Eddie
"""

import torch.nn as nn
import torch
import numpy as np
from function import bboxes_iou
import torch.nn.functional as F

class Detection_loss(nn.Module):
    def __init__(self, n_classes, n_anchors, image_size, device, batch):
        super(Detection_loss, self).__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.anchors = [[13, 16], [28, 32], [62, 35]]# [47, 72], [109, 62], [88, 126], [189, 105], [165, 230], [355, 191]]
        self.anchor_masks = [[0, 1, 2]]#, [3, 4, 5], [6, 7, 8]]
        self.device = device
        self.strides = [8]#, 16, 32]
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []
        self.ignore_thre = 0.5
        self.image_size = image_size
        
        for i in range(len(self.strides)):
            all_anchors_grid = [(w / image_size / self.strides[i], h / image_size / self.strides[i]) for w, h in self.anchors]
            print(all_anchors_grid)
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anchor_masks[i]], dtype = np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype = np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype = np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype = torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype = torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(device)
            #print(anchor_w)
            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)
            
    def prepare_target(self, pred, labels, batchsize, fsize, n_ch, layer_id):
        # target assignment
        target_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        target_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)
        for b in range(batchsize):
            n = labels[b].shape[0]
            if n == 0:
                continue
            label_x = (labels[b][..., 0] + labels[b][..., 2]) / (self.strides[layer_id] * 2)
            label_y = (labels[b][..., 1] + labels[b][..., 3]) / (self.strides[layer_id] * 2)
            label_w = (labels[b][..., 2]) / self.strides[layer_id]
            label_h = (labels[b][..., 3]) / self.strides[layer_id]
            
            label_i = label_x.to(torch.int16).cpu().numpy()
            label_j = label_y.to(torch.int16).cpu().numpy()
            
            label_box = torch.zeros(n, 4).to(self.device)
            label_box[:, 2] = label_w
            label_box[:, 3] = label_h
            anchor_ious_all = bboxes_iou(label_box.cpu(), self.ref_anchors[layer_id], CIoU = True)
            best_n_all = anchor_ious_all.argmax(dim = 1)
            best_n = best_n_all % self.n_anchors
            
            best_n_mask = ((best_n_all == self.anchor_masks[layer_id][0]) |
                           (best_n_all == self.anchor_masks[layer_id][1]) |
                           (best_n_all == self.anchor_masks[layer_id][2]))

            if sum(best_n_mask) == 0:
                continue
            
            label_box[:, 0] = label_x
            label_box[:, 1] = label_y
            pred_ious = bboxes_iou(pred[b].view(-1, 4), label_box)
            pred_best_ious, _ = pred_ious.max(dim = 1)
            pred_best_ious = (pred_best_ious > self.ignore_thre)
            pred_best_ious = pred_best_ious.view(pred[b].shape[:3])
            #print("pred_best_ious : ", pred_best_ious.shape)
            obj_mask[b] = ~pred_best_ious
            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = label_i[ti], label_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    target_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = label_x[ti] - label_x[ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = label_x[ti] - label_x[ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                            label_w[ti] / torch.Tensor(self.masked_anchors[layer_id])[best_n[ti], 0] + 1e-16
                        )
                    target[b, a, j, i, 3] = torch.log(
                            label_h[ti] / torch.Tensor(self.masked_anchors[layer_id])[best_n[ti], 1] + 1e-16
                        )
                    target[b, a, j, i, 4] = 1
                    #print(target[b, a, j, i, 5:5 + 12])
                    target[b, a, j, i, 5 + labels[b][ti, 4].to(torch.int16).cpu().numpy()] = 1
# =============================================================================
#                     print(b, a, j, i, labels[b][ti, 4])
#                     print(target[b, a, j, i, 5:5 + 12])
#                     print()
# =============================================================================
                    target_scale[b, a, j, i, :] = torch.sqrt(2 - label_w[ti] * label_h[ti] / fsize / fsize)
                    
        return obj_mask, target_mask, target_scale, target
        
        
    def forward(self, x, label = None):
        loss, loss_wh, loss_xy, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for layer_id in range(len(self.anchor_masks)):
            output = x[layer_id]
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_cha = 5 + self.n_classes
            
            output = output.view(batchsize, self.n_anchors, n_cha, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)
            
            # logistic activation for xy, obj, cls 
            output[..., np.r_[:2, 4:n_cha]] = torch.sigmoid(output[..., np.r_[:2, 4:n_cha]])
            pred_bbox = output[..., :4].clone()
            pred_bbox[..., 0] += self.grid_x[layer_id]
            pred_bbox[..., 1] += self.grid_y[layer_id]
            
            pred_bbox[..., 2] = torch.exp(pred_bbox[..., 2] * self.anchor_w[layer_id])
            pred_bbox[..., 3] = torch.exp(pred_bbox[..., 3] * self.anchor_h[layer_id])
            obj_mask, target_mask, target_scale, target = self.prepare_target(pred_bbox, label, batchsize, fsize, n_cha, layer_id)
            
            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_cha]] *= target_mask
            output[..., 2:4] *= target_scale
            
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_cha]] *= target_mask
            target[..., 2:4] *= target_scale
            
            loss_xy += F.binary_cross_entropy(input = output[..., :2], target = target[..., :2],
                                              weight = target_scale * target_scale, reduction = "sum")
            loss_wh += F.mse_loss(input = output[..., 2:4], target = target[..., 2:4], reduction = 'sum') / 2
            loss_obj += F.binary_cross_entropy(input = output[..., 4], target = target[..., 4], reduction = 'sum')
            loss_cls += F.binary_cross_entropy(input = output[..., 5:], target = target[..., 5:], reduction = 'sum')
            loss_l2 += F.mse_loss(input = output, target = target, reduction = 'sum')
            
        loss = loss_xy + loss_wh + loss_obj + loss_cls
            
        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2
    
if __name__ == '__main__':
    loss = Detection_loss(3, 3, 416, "cpu", batch = 2)
