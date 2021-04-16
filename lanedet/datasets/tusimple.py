import os.path as osp
import numpy as np
import cv2
import torchvision
import lanedet.utils.transforms as tf
from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, img_path, data_list, cfg=None):
        super().__init__(img_path, data_list, 'seg_label/list', cfg)

    def transform_train(self):
        input_mean = self.cfg.img_norm['mean']
        train_transform = torchvision.transforms.Compose([
            tf.GroupRandomRotation(),
            tf.GroupRandomHorizontalFlip(),
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return train_transform


    def init(self):
        with open(osp.join(self.list_path, self.data_list)) as f:
            for line in f:
                line_split = line.strip().split(" ")
                self.img_name_list.append(line_split[0])
                self.full_img_path_list.append(self.img_path + line_split[0])
                self.label_list.append(self.img_path + line_split[1])
                self.exist_list.append(
                    np.array([int(line_split[2]), int(line_split[3]),
                              int(line_split[4]), int(line_split[5]),
                              int(line_split[6]), int(line_split[7])
                              ]))

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end+1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i+1] < 0]
                gap_end = [i+1 for i,
                           x in enumerate(lane[:-1]) if x < 0 and lane[i+1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                gap_end[i] - id) / gap_width * lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end+1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_coord(self, prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)
    
        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = resize_shape
        H -= self.cfg.cut_height
    
        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)

        return coords

    def probmap2lane(self, seg_pred, exist, resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold
    
        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []
    
        for i in range(self.cfg.num_classes - 1):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_coord(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
    
    
        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        #print(coordinates)
    
        return coordinates

    def get_lane(self, output):
        seg_pred, exist_pred = output['seg'], output['exist']
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        rets = []
        for seg, exist in zip(seg_pred, exist_pred):
            lanes = self.probmap2lane(seg, exist)
            rets.append(lanes)
        return rets

