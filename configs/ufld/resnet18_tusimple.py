net = dict(
    type='Segmentor',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)
featuremap_out_channel = 512

aggregator=None

griding_num = 100
num_classes = 6
heads = [
    dict(type='LaneCls',
        dim = (griding_num + 1, 56, num_classes))
]

trainer = dict(
    type='LaneCls'
)

evaluator = dict(
    type='Tusimple',
)

optimizer = dict(
  type = 'sgd',
  lr = 0.030,
  weight_decay = 1e-4,
  momentum = 0.9
)

import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

optimizer = dict(
  type = 'sgd',
  lr = 0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 100
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

dataset_path = './data/tusimple'
dataset = dict(
    train=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='seg_label/list/train_val_gt.txt'
    ),
    val=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='seg_label/list/test_gt.txt'
    ),
    test=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='seg_label/list/test_gt.txt'
    )
)

workers = 12
ignore_label = 255
log_interval = 100
eval_ep = 1
save_ep = epochs
row_anchor='tusimple_row_anchor'
test_json_file='data/tusimple/test_label.json'
y_pixel_gap = 10
