import torch.nn as nn
import torch
import torch.nn.functional as F
from lanedet.runner.utils.logger import get_logger

from lanedet.runner.registry import EVALUATOR 
import json
import os
import cv2

from .lane import LaneEval

def split_path(path):
    """split path tree into list"""
    folders = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folders.insert(0, folder)
        else:
            if path != "":
                folders.insert(0, path)
            break
    return folders


@EVALUATOR.register_module
class Tusimple(nn.Module):
    def __init__(self, cfg):
        super(Tusimple, self).__init__()
        self.cfg = cfg 
        exp_dir = os.path.join(self.cfg.work_dir, "output")
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)
        self.out_path = os.path.join(exp_dir, "coord_output")
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        self.dump_to_json = [] 
        self.logger = get_logger('lanedet')
        if cfg.view:
            self.view_dir = os.path.join(self.cfg.work_dir, 'vis')

    def evaluate_lane(self, dataset, res, batch):
        img_name = batch['meta']['img_name']
        img_path = batch['meta']['full_img_path']
        for b in range(len(res)):
            lane_coords = res[b]
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])

            path_tree = split_path(img_name[b])
            save_dir, save_name = path_tree[-3:-1], path_tree[-1]
            save_dir = os.path.join(self.out_path, *save_dir)
            save_name = save_name[:-3] + "lines.txt"
            save_name = os.path.join(save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(save_name, "w") as f:
                for l in lane_coords:
                    for (x, y) in l:
                        print("{} {}".format(x, y), end=" ", file=f)
                    print(file=f)

            json_dict = {}
            json_dict['lanes'] = []
            json_dict['h_sample'] = []
            json_dict['raw_file'] = os.path.join(*path_tree[-4:])
            json_dict['run_time'] = 0
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_dict['lanes'].append([])
                for (x, y) in l:
                    json_dict['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_dict['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_dict))
            
            if self.cfg.view:
                img = cv2.imread(img_path[b])
                new_img_name = img_name[b].replace('/', '_')
                save_dir = os.path.join(self.view_dir, new_img_name)
                dataset.view(img, lane_coords, save_dir)

    def evaluate(self, dataset, output, batch):
        res = dataset.get_lane(output)
        self.evaluate_lane(dataset, res, batch)

    def summarize(self):
        best_acc = 0
        output_file = os.path.join(self.out_path, 'predict_test.json')
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_result, acc = LaneEval.bench_one_submit(output_file,
                            self.cfg.test_json_file)

        self.logger.info(eval_result)
        self.dump_to_json = []
        best_acc = max(acc, best_acc)
        return best_acc
