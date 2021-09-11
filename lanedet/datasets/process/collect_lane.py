from functools import cmp_to_key
import random
import copy
import PIL.Image
import PIL.ImageDraw

import cv2
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import math

from ..registry import PROCESS 
from mmcv.parallel import DataContainer as DC
from .transforms import to_tensor

def get_line_intersection(x, y, line):
    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if val >= s and val <= e and s != e:
            return True
        else:
            return False

    def choose_min_reg(val, ref):
        min_val = 1e5
        index = -1
        if len(val) == 0:
            return None
        else:
            for i, v in enumerate(val):
                if abs(v - ref) < min_val:
                    min_val = abs(v - ref)
                    index = i
        return val[index]

    reg_y = []
    reg_x = []

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(x, point_start[0], point_end[0]):
            k = (point_end[1] - point_start[1]) / (
                point_end[0] - point_start[0])
            reg_y.append(k * (x - point_start[0]) + point_start[1])
    reg_y = choose_min_reg(reg_y, y)

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (
                point_end[1] - point_start[1])
            reg_x.append(k * (y - point_start[1]) + point_start[0])
    reg_x = choose_min_reg(reg_x, x)
    return reg_x, reg_y

def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[1]
    end = line[0]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx = dx / norm
    dy = dy / norm
    extend_point = (start[0] + dx * dis, start[1] + dy * dis)
    extended.insert(0, extend_point)
    return extended

def select_mask_points(ct, r, shape, max_sample=5):

    def in_range(pt, w, h):
        if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
            return True
        else:
            return False

    h, w = shape[:2]
    valid_points = []
    r = max(int(r // 2), 1)
    start_x, end_x = ct[0] - r, ct[0] + r
    start_y, end_y = ct[1] - r, ct[1] + r
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            if x == ct[0] and y == ct[1]:
                continue
            if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                valid_points.append([x, y])
    if len(valid_points) > max_sample - 1:
        valid_points = random.sample(valid_points, max_sample - 1)
    valid_points.append([ct[0], ct[1]])
    return valid_points

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy

def draw_label(mask,
               polygon_in,
               val,
               shape_type='polygon',
               width=3,
               convert=False):
    polygon = copy.deepcopy(polygon_in)
    mask = PIL.Image.fromarray(mask)
    xy = []
    if convert:
        for i in range(len(polygon) // 2):
            xy.append((polygon[2 * i], polygon[2 * i + 1]))
    else:
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

    if shape_type == 'polygon':
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
    else:
        PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
    mask = np.array(mask, dtype=np.uint8)
    return mask

def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):

            pts = list(I.coords)
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None

def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


@PROCESS.register_module
class CollectLane(object):
    def __init__(
            self,
            down_scale,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg'),
            hm_down_scale=None,
            line_width=3,
            max_mask_sample=5,
            perspective=False,
            radius=2,
            cfg=None,
    ):
        self.meta_keys = meta_keys
        self.keys = keys
        self.cfg = cfg
        self.down_scale = down_scale
        self.hm_down_scale = hm_down_scale if hm_down_scale is not None else down_scale
        self.line_width = line_width
        self.max_mask_sample = max_mask_sample
        self.radius = radius

    def target(self, results):
        def min_dis_one_point(points, idx):
            min_dis = 1e6
            for i in range(len(points)):
                if i == idx:
                    continue
                else:
                    d = cal_dis(points[idx], points[i])
                    if d < min_dis:
                        min_dis = d
            return min_dis

        output_h = self.cfg.img_height
        output_w = self.cfg.img_width
        mask_h = int(output_h // self.down_scale)
        mask_w = int(output_w // self.down_scale)
        hm_h = int(output_h // self.hm_down_scale)
        hm_w = int(output_w // self.hm_down_scale)
        results['hm_shape'] = [hm_h, hm_w]
        results['mask_shape'] = [mask_h, mask_w]
        ratio_hm_mask = self.down_scale / self.hm_down_scale

        # gt init
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)
        gt_masks = []

        # gt heatmap and ins of bank
        gt_points = results['gt_points']
        # gt_points = results['lanes']
        valid_gt = []
        for pts in gt_points:
            id_class = 1
            pts = convert_list(pts, self.down_scale)
            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))
            pts = clamp_line(
                pts, box=[0, 0, mask_w - 1, mask_h - 1], min_length=1)
            if pts is not None and len(pts) > 1:
                valid_gt.append([pts, id_class - 1])

        # draw gt_hm_lane
        gt_hm_lane_ends = []
        for l in valid_gt:
            label = l[1]
            point = (l[0][0][0] * ratio_hm_mask, l[0][0][1] * ratio_hm_mask)
            gt_hm_lane_ends.append([point, l[0]])
        radius = [self.radius for _ in range(len(gt_hm_lane_ends))]


        for (end_point, line), r in zip(gt_hm_lane_ends, radius):
            pos = np.zeros((mask_h), np.float32)
            pos_mask = np.zeros((mask_h), np.float32)
            pt_int = [int(end_point[0]), int(end_point[1])]
            draw_umich_gaussian(gt_hm[0], pt_int, r)
            line_array = np.array(line)
            y_min, y_max = int(np.min(line_array[:, 1])), int(
                np.max(line_array[:, 1]))
            mask_points = select_mask_points(
                pt_int, r, (hm_h, hm_w), max_sample=self.max_mask_sample)
            reg = np.zeros((1, mask_h, mask_w), np.float32)
            reg_mask = np.zeros((1, mask_h, mask_w), np.float32)

            extended_line = extend_line(line)
            line_array = np.array(line)
            y_min, y_max = np.min(line_array[:, 1]), np.max(line_array[:, 1])
            # regression
            m = np.zeros((mask_h, mask_w), np.uint8)
            lane_range = np.zeros((1, mask_h), np.int64)
            line_array = np.array(line)

            polygon = np.array(extended_line)
            polygon_map = draw_label(
                m, polygon, 1, 'line', width=self.line_width + 9) > 0
            for y in range(polygon_map.shape[0]):
                for x in np.where(polygon_map[y, :])[0]:
                    reg_x, _ = get_line_intersection(x, y, line)
                    # kps and kps_mask:
                    if reg_x is not None:
                        offset = reg_x - x
                        reg[0, y, x] = offset
                        if abs(offset) < 10:
                            reg_mask[0, y, x] = 1
                        if y >= y_min and y <= y_max:
                            pos[y] = reg_x
                            pos_mask[y] = 1
                        lane_range[:, y] = 1


            gt_masks.append({
                'reg': reg,
                'reg_mask': reg_mask,
                'points': mask_points,
                'row': pos,
                'row_mask': pos_mask,
                'range': lane_range,
                'label': 0
            })

        results['gt_hm'] = gt_hm
        results['gt_masks'] = gt_masks
        results['down_scale'] = self.down_scale
        results['hm_down_scale'] = self.hm_down_scale
        return True

    def __call__(self, results):
        data = {}
        img_meta = {}

        valid = self.target(results)
        if not valid:
            return None
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

