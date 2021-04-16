"""
Reference:
    https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import torch
from PIL import Image
import os
import os.path as osp
import pdb
import numpy as np
import cv2
import torchvision.transforms as transforms
from lanedet.utils import mytransforms as mytransforms
from lanedet.utils.mytransforms import find_start_pos
import scipy
from .base_dataset import BaseDataset


from .registry import DATASETS

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

def loader_func(path):
    return Image.open(path)

@DATASETS.register_module
class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, data_list, cfg=None):
        super(LaneClsDataset, self).__init__()
        self.cfg = cfg
        self.img_transform = None
        self.simu_transform = None
        self.path = img_path
        self.griding_num = cfg.griding_num
        #self.use_aux = cfg.use_aux
        self.ori_img_h = cfg.ori_img_h
        self.ori_img_w = cfg.ori_img_w

        self.is_training = not ('test' in data_list)

        list_path = os.path.join(img_path, data_list)
        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = eval(cfg.row_anchor)
        self.row_anchor.sort()
        self.set_transform()

    def view(self, img, coords, file_path=None):
        for coord in coords:
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

        if file_path is not None:
            if not os.path.exists(osp.dirname(file_path)):
                os.makedirs(osp.dirname(file_path))
            cv2.imwrite(file_path, img)

    def set_transform(self):
        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if self.is_training:
            self.simu_transform = mytransforms.Compose2([
                mytransforms.RandomRotate(6),
                mytransforms.RandomUDoffsetLABEL(100),
                mytransforms.RandomLROffsetLABEL(200)
                ])


    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        if self.is_training:
            img_name, label_name = l_info[0], l_info[1]
        else:
            img_name = l_info[0]
        
        img_path = self.path +'/'+ img_name
        img = loader_func(img_path)

        cls_label = None
        if self.is_training:
            label_path = self.path + label_name
            label = loader_func(label_path)
            if self.simu_transform is not None:
                img, label = self.simu_transform(img, label)

            lane_pts = self._get_index(label)

            w, h = img.size
            cls_label = self._grid_pts(lane_pts, self.griding_num, w)

        if self.img_transform is not None:
            img = self.img_transform(img)

        # if self.use_aux:
        #     assert self.segment_transform is not None
        #     seg_label = self.segment_transform(label)
        meta = {'img_name': img_name, 'full_img_path': img_path}

        if self.is_training:
            ret = {'img': img, 'cls_label': cls_label, 'meta': meta}
        else:
            ret = {'img': img, 'meta': meta}

        return ret


    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        tot_len = col_sample[1] - col_sample[0]
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // tot_len) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def postprocess(self, out, localization_type='rel', flip_updown=True):
        predictions = []
        griding_num = self.cfg.griding_num
        for j in range(out.shape[0]):
            out_j = out[j].data.cpu().numpy()
            if flip_updown:
                out_j = out_j[:, ::-1, :]
            if localization_type == 'abs':
                out_j = np.argmax(out_j, axis=0)
                out_j[out_j == griding_num] = -1
                out_j = out_j + 1
            elif localization_type == 'rel':
                prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
                idx = np.arange(griding_num) + 1
                idx = idx.reshape(-1, 1, 1)
                loc = np.sum(prob * idx, axis=0)
                out_j = np.argmax(out_j, axis=0)
                loc[out_j == griding_num] = 0
                out_j = loc
            else:
                raise NotImplementedError
            predictions.append(out_j)
        return predictions
    
    def get_lane(self, pred):
        predictions = self.postprocess(pred['cls']) 
        ret = []
        griding_num = self.cfg.griding_num
        for out in predictions:
            lanes = []
            for i in range(out.shape[1]):
                if sum(out[:, i] != 0) <= 2: continue
                out_i = out[:, i]
                coord = np.zeros((out.shape[0], 2))
                coord.fill(-1)
                for k in range(out.shape[0]):
                    coord[k][0] = int((out_i[k]-0.5) * self.ori_img_w / (griding_num - 1)) if out[k, i] > 0 else -1
                    coord[k][1] = int(self.ori_img_h-k*self.cfg.y_pixel_gap) - 10
                lanes.append(coord)
            ret.append(lanes)
        return ret

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))

        num_classes = self.cfg.num_classes

        all_idx = np.zeros((num_classes, len(sample_tmp),2))
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, num_classes+1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        all_idx_cp = all_idx.copy()
        for i in range(num_classes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue

            valid = all_idx_cp[i,:,1] != -1
            valid_idx = all_idx_cp[i,valid,:]
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                continue
            if len(valid_idx) < 6:
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
