# RESA: Recurrent Feature-Shift Aggregator for Lane Detection

## Introduction

```latex
@misc{zheng2020resa,
      title={RESA: Recurrent Feature-Shift Aggregator for Lane Detection}, 
      author={Tu Zheng and Hao Fang and Yi Zhang and Wenjian Tang and Zheng Yang and Haifeng Liu and Deng Cai},
      year={2020},
      eprint={2008.13719},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Models

| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| RESA        | ResNet50 | CULane |F1: 75.92| [config](https://github.com/Turoad/lanedet/blob/main/configs/resa/resa50_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/resa_r50_culane.zip)|
| RESA | ResNet34 | CULane |F1: 75.85| [config](https://github.com/Turoad/lanedet/blob/main/configs/resa/resa34_culane.py)  |[model](https://github.com/Turoad/lanedet/releases/download/1.0/resa_r34_culane.zip)|
| RESA | ResNet34 | Tusimple |acc: 96.86| [config](https://github.com/Turoad/lanedet/blob/main/configs/resa/resa34_tusimple.py)  |[model](https://github.com/Turoad/lanedet/releases/download/1.0/resa_r34_tusimple.zip)|
| RESA | ResNet18 | Tusimple |acc: 96.73| [config](https://github.com/Turoad/lanedet/blob/main/configs/resa/resa18_tusimple.py)  |[model](https://github.com/Turoad/lanedet/releases/download/1.0/resa_r18_tusimple.zip)|
