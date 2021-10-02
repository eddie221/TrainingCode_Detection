# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:24:45 2021

@author: Eddie
"""

from function import draw_bbox
from transform import *
from dataset import Detection_dataset, collate
from network import Detector
import torch
from loss import Detection_loss
from torch.utils.data import DataLoader
import tqdm

def get_anchor_box(outputs, n_anchors, anchor_sizes, anchor_masks, anchor_strides, n_classes):
    print("get_anchor_box")
    for layer_id, output in enumerate(outputs):
        batches = output.shape[0]
        output = output.detach().cpu().numpy()
        
        # create basic grid
        grid_x, grid_y = np.meshgrid(np.linspace(0, output.shape[3] - 1, output.shape[3]), np.linspace(0, output.shape[3] - 1, output.shape[3]))
        grid_x = np.expand_dims(np.expand_dims(grid_x, axis = 0), axis = 0)
        grid_y = np.expand_dims(np.expand_dims(grid_y, axis = 0), axis = 0)
        
        # prepare bbox
        output = output.reshape(batches, n_anchors, n_classes + 5, output.shape[-2], output.shape[-1])
        output[:, :, np.r_[:2, 4:n_classes + 5]] = 1 / (1 + np.exp(-output[:, :, np.r_[:2, 4:n_classes + 5]]))
        output[:, :, 0] = output[:, :, 0] + grid_x
        output[:, :, 1] = output[:, :, 1] + grid_y
        print(output.shape)
        
        confs = (1 + np.exp(-output[:, :, 4]))
        bboxes = np.transpose(output[:, :, :4], (0, 1, 3, 4, 2))
        confs_select = np.where(confs > 0.5)
        bboxes_select = bboxes[confs_select[0], confs_select[1], confs_select[2], confs_select[3]]
        print()

def training(model, img_norm, anno, loss_func, optimizer, phase):
    outputs = model(img_norm.cuda())
    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = loss_func(outputs, anno)
    
    if phase == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    for output in outputs:
        output = output.detach().cpu()
    return outputs, loss.item(), loss_xy.item(), loss_wh.item(), loss_obj.item(), loss_cls.item(), loss_l2.item()
    
def load_param(model):
    param = torch.load("./best.pkl")
    model.load_state_dict(param)
    return model

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count

if __name__ == '__main__':
    transform = Compose([Resize((416, 416)), RandomFlip(0.1)])
    dataset = Detection_dataset("../dataset/VisDrone2019-DET-train/images", "../dataset/VisDrone2019-DET-train/annotations", transform)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, collate_fn = collate)
    #img_norm, anno, img = dataset[100]
    model = Detector(num_classes = 12, n_anchors = 3).cuda()
    model = load_param(model)
    model.eval()
    detection_loss = Detection_loss(12, 3, 416, "cuda:0", 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    loss = 0
    min_loss = 100
    for epoch in range(1):
        for phase in ["val"]:
            lossMeter = AverageMeter()
            lossMeter_xy = AverageMeter()
            lossMeter_wh = AverageMeter()
            lossMeter_obj = AverageMeter()
            lossMeter_cls = AverageMeter()
            lossMeter_l2 = AverageMeter()
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
            
            for img_norm, anno, img in tqdm.tqdm(dataloader, desc = f"Epoch {epoch + 1} / 100"):#dataloader:
                outputs, loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = training(model, img_norm, anno, detection_loss, optimizer, phase)
                lossMeter.update(loss, img_norm.shape[0])
                get_anchor_box(outputs, 3, [[13, 16], [28, 32], [62, 35]], [[0, 1, 2]], [8], 12)
# =============================================================================
#                 print("Loss : ", loss)
#                 print("anno : ", anno[0].shape)
#                 print(anno[0][0])
# =============================================================================
                break
            break
        break