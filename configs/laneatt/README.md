# Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection

## Introduction

```latex
@InProceedings{tabelini2021cvpr,
  author    = {Lucas Tabelini
               and Rodrigo Berriel
               and Thiago M. Paix\~ao
               and Claudine Badue
               and Alberto Ferreira De Souza
               and Thiago Oliveira-Santos},
  title     = {{Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection}},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```

## Models

| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| LaneATT | ResNet18 | CULane | F1: 75.10 | [config](https://github.com/Turoad/lanedet/blob/main/configs/laneatt/resnet18_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/laneatt_r18_culane.zip)|
|LaneATT | MobileNetV2 | CULane |F1: 74.11 | [config](https://github.com/Turoad/lanedet/blob/main/configs/laneatt/mobilenetv2_culane.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/laneatt_mobilenetv2_culane.zip)|
| LaneATT | ResNet18 | TuSimple|acc: 95.66| [config](https://github.com/Turoad/lanedet/blob/main/configs/laneatt/resnet18_tusimple.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/laneatt_r18_tusimple.zip)|
|LaneATT | ResNet34 | TuSimple|acc: 95.81 | [config](https://github.com/Turoad/lanedet/blob/main/configs/laneatt/resnet34_tusimple.py)  | [model](https://github.com/Turoad/lanedet/releases/download/1.0/laneatt_r34_tusimple.zip)|


