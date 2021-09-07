# Spatial As Deep: Spatial CNN for Traffic Scene Understanding 

## Introduction

```latex
@inproceedings{pan2018SCNN,  
  author = {Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang},  
  title = {Spatial As Deep: Spatial CNN for Traffic Scene Understanding},  
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},  
  month = {February},  
  year = {2018}  
}
```

## Models
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| SCNN       | ResNet50 | CULane | F1: 74.89| [config](https://github.com/Turoad/lanedet/blob/main/configs/scnn/resa18_culane.py)  |[model](https://github.com/Turoad/lanedet/releases/download/1.0/scnn_r50_culane.zip) |
| SCNN       | ResNet18 | Tusimple |acc: 96.05| [config](https://github.com/Turoad/lanedet/blob/main/configs/scnn/resa18_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/scnn_r18_tusimple.zip)|
