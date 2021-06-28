net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)
featuremap_out_channel = 512

griding_num = 100
num_classes = 6
heads = dict(type='LaneCls',
        dim = (griding_num + 1, 56, num_classes))

trainer = dict(
    type='LaneCls'
)

evaluator = dict(
    type='Tusimple',
)

import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

optimizer = dict(
  type = 'SGD',
  lr = 0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 150
batch_size = 4
total_iter = (3616 // batch_size + 1) * epochs 
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)


img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

ori_img_h = 720
ori_img_w = 1280
img_h = 288
img_w = 800
cut_height=0
sample_y = range(710, 150, -10)

dataset_type = 'TuSimple'
dataset_path = './data/tusimple'
row_anchor = 'tusimple_row_anchor'

train_process = [
    dict(type='RandomRotation', degree=(-6, 6)),
    dict(type='RandomUDoffsetLABEL', max_offset=100),
    dict(type='RandomLROffsetLABEL', max_offset=200),
    dict(type='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_classes=num_classes),
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img', 'cls_label']),
]

val_process = [
    dict(type='Resize', size=(img_w, img_h)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)


workers = 12
ignore_label = 255
log_interval = 100
eval_ep = 1
save_ep = epochs
row_anchor='tusimple_row_anchor'
test_json_file='data/tusimple/test_label.json'
lr_update_by_epoch = False
