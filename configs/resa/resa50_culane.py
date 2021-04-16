net = dict(
    type='Segmentor',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet50',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    in_channels=[64, 128, 256, 512]
)
featuremap_out_channel = 128
featuremap_out_stride = 8

aggregator = dict(
    type='RESA',
    direction=['d', 'u', 'r', 'l'],
    alpha=2.0,
    iter=4,
    conv_stride=9,
)

heads = [
    dict(type='PlainDecoder'),
    dict(type='ExistHead'),
]

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
  type = 'sgd',
  lr = 0.030,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 12
batch_size = 8
total_iter = (88880 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

seg_loss_weight = 1.0
eval_ep = 6
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 288
img_width = 800
cut_height = 240 

dataset_path = './data/CULane'
dataset = dict(
    train=dict(
        type='CULane',
        img_path=dataset_path,
        data_list='train_gt.txt',
    ),
    val=dict(
        type='CULane',
        img_path=dataset_path,
        data_list='test_img.txt',
    ),
    test=dict(
        type='CULane',
        img_path=dataset_path,
        data_list='test_img.txt',
    )
)


workers = 12
num_classes = 4 + 1
ignore_label = 255
log_interval = 1000

