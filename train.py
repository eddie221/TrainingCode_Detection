# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:54:49 2021

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
    dataloader = DataLoader(dataset, batch_size = 2, shuffle = False, collate_fn = collate)
    #img_norm, anno, img = dataset[100]
    model = Detector(num_classes = 12, n_anchors = 3).cuda()
    #model = load_param(model)
    detection_loss = Detection_loss(12, 3, 416, "cuda:0", 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    loss = 0
    min_loss = 100
    for epoch in range(5):
        for phase in ["train", "val"]:
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
                print("Loss : ", loss)
                break
            break
        break
                
# =============================================================================
#             if lossMeter.avg < min_loss:
#                 min_loss = lossMeter.avg
#                 torch.save(model.state_dict(), "./best.pkl")
# =============================================================================

