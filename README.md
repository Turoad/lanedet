# LaneDet
## Introduction
LaneDet is an open source lane detection toolbox based on PyTorch that aims to pull together a wide variety of state-of-the-art lane detection models. Developers can reproduce these SOTA methods and build their own methods.

![demo image](.github/_clips_0601_1494452613491980502_20.jpg)

## Table of Contents
* [Introduction](#Introduction)
* [Benchmark and model zoo](#Benchmark-and-model-zoo)
* [Installation](#Installation)
* [Getting Started](#Getting-started)
* [Contributing](#Contributing)
* [Licenses](#Licenses)
* [Acknowledgement](#Acknowledgement)

## Benchmark and model zoo
Supported backbones:
- [x] ResNet
- [x] ERFNet
- [x] VGG
- [x] MobileNet

Supported detectors:
- [x] [SCNN](configs/scnn)
- [x] [UFLD](configs/ufld)
- [x] [RESA](configs/resa)
- [x] [LaneATT](configs/laneatt)
- [x] [CondLane](configs/condlane)


## Installation
<!--
Please refer to [INSTALL.md](INSTALL.md) for installation.
-->

### Clone this repository
```
git clone https://github.com/turoad/lanedet.git
```
We call this directory as `$LANEDET_ROOT`

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n lanedet python=3.8 -y
conda activate lanedet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch torchvision

# Install python packages
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $RESA_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $RESA_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

## Getting Started
### Training

For training, run

```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_ids]
```


For example, run
```Shell
python main.py configs/resa/resa50_culane.py --gpus 0 1 2 3
```

### Testing
For testing, run
```Shell
python main.py [configs/path_to_your_config] --validate --load_from [path_to_your_model] [gpu_num]
```

For example, run
```Shell
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0 1 2 3
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

For example, run
```Shell
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0 --view
```

### Inference
See `tools/detect.py` for detailed information.
```
python tools/detect.py --help

usage: detect.py [-h] [--img IMG] [--show] [--savedir SAVEDIR]
                 [--load_from LOAD_FROM]
                 config

positional arguments:
  config                The path of config file

optional arguments:
  -h, --help            show this help message and exit
  --img IMG             The path of the img (img file or img_folder), for
                        example: data/*.png
  --show                Whether to show the image
  --savedir SAVEDIR     The root of save directory
  --load_from LOAD_FROM
                        The path of model
```
To run inference on example images in `./images` and save the visualization images in `vis` folder:
```
python tools/detect.py configs/resa/resa34_culane.py --img images\
          --load_from resa_r34_culane.pth --savedir ./vis
```


## Contributing
We appreciate all contributions to improve LaneDet.  Any pull requests or issues are welcomed.

## Licenses
This project is released under the [Apache 2.0 license](LICNESE).


## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
* [XingangPan/SCNN](https://github.com/XingangPan/SCNN)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->

<!-- 
## Citation
If you use
```
@misc{zheng2021lanedet,
  author =       {Tu Zheng},
  title =        {LaneDet},
  howpublished = {\url{https://github.com/turoad/lanedet}},
  year =         {2021}
}
``` -->
