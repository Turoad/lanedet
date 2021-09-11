# CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution

## Introduction

```latex
@article{liu2021condlanenet,
  title={CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution},
  author={Liu, Lizhe and Chen, Xiaohao and Zhu, Siyu and Tan, Ping},
  journal={arXiv preprint arXiv:2105.05003},
  year={2021}
}
```

## Models
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| CondLane   | ResNet101 | CULane | F1: 79.47| [config](configs/condlane/resnet101_culane.py)  |[model](https://github.com/Turoad/lanedet/releases/download/1.0/condlane_r101_culane.pth.zip) |
