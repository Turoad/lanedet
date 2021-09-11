# Ultra Fast Structure-aware Deep Lane Detection

## Introduction

```latex
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

## Models
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| UFLD | ResNet18 | CULane |F1: 69.47| [config](https://github.com/Turoad/lanedet/blob/main/configs/ufld/resa18_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/ufld_r18_culane.zip)|
| UFLD | ResNet18 | Tusimple |acc: 95.86| [config](https://github.com/Turoad/lanedet/blob/main/configs/ufld/resa18_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/ufld_r18_tusimple.zip)|
