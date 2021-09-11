net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 128
featuremap_out_stride = 8
sample_y = range(589, 230, -20)

aggregator = dict(
    type='RESA',
    direction=['d', 'u', 'r', 'l'],
    alpha=2.0,
    iter=4,
    conv_stride=9,
)

heads = dict( 
    type='LaneSeg',
    decoder=dict(type='PlainDecoder'),
    exist=dict(type='ExistHead'),
    thr=0.3,
    sample_y=sample_y,
)

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
  type = 'SGD',
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
ori_img_h = 590
ori_img_w = 1640

train_process = [
    dict(type='RandomRotation', degree=(-2, 2)),
    dict(type='RandomHorizontalFlip'),
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'mask', 'lane_exist']),
]

val_process = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/CULane'
dataset = dict(
    train=dict(
        type='CULane',
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)


workers = 12
num_classes = 4 + 1
ignore_label = 255
log_interval = 1000

lr_update_by_epoch = False
