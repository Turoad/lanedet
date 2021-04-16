import os
import os.path as osp
import numpy as np
import torchvision
import lanedet.utils.transforms as tf
from .base_dataset import BaseDataset
from .registry import DATASETS
import cv2
import torch


@DATASETS.register_module
class CULane(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, cfg=cfg)
        self.ori_imgh = 590
        self.ori_imgw = 1640

    def init(self):
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                self.img_name_list.append(line_split[0])
                self.full_img_path_list.append(self.img_path + line_split[0])
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5])]))

    def transform_train(self):
        train_transform = torchvision.transforms.Compose([
            tf.GroupRandomRotation(degree=(-2, 2)),
            tf.GroupRandomHorizontalFlip(),
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return train_transform

    def get_lane(self, output):
        segs, exists = output['seg'], output['exist']
        segs = segs.cpu().numpy()
        exists = exists.cpu().numpy()
        ret = []
        for seg, exist in zip(segs, exists):
            lanes = self.probmap2lane(seg, exist)
            ret.append(lanes)
        return ret


    def probmap2lane(self, probmaps, exists, pts=18):
        coords = []
        probmaps = probmaps[1:, ...]
        exists = exists > 0.5
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            thr = 0.3
            coordinate = np.zeros(pts)
            cut_height = self.cfg.cut_height
            for i in range(pts):
                line = probmap[round(
                    self.cfg.img_height-i*20/(self.ori_imgh-cut_height)*self.cfg.img_height)-1]

                if np.max(line) > thr:
                    coordinate[i] = np.argmax(line)+1
            if np.sum(coordinate > 0) < 2:
                continue
    
            img_coord = np.zeros((pts, 2))
            img_coord[:, :] = -1
            for idx, value in enumerate(coordinate):
                if value > 0:
                    img_coord[idx][0] = round(value*self.ori_imgw/self.cfg.img_width-1)
                    img_coord[idx][1] = round(self.ori_imgh-idx*20-1)
    
            img_coord = img_coord.astype(int)
            coords.append(img_coord)
    
        return coords
