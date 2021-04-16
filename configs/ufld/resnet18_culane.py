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

griding_num = 200
num_classes = 4
heads = [
    dict(type='LaneCls',
        dim = (griding_num + 1, 18, num_classes))
]

trainer = dict(
    type='LaneCls'
)

evaluator = dict(
    type='CULane',
)

optimizer = dict(
  type = 'sgd',
  lr = 0.015,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 50
batch_size = 32
total_iter = (88880 // batch_size + 1) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)


img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

ori_img_h = 590 
ori_img_w = 1640 
img_h = 288
img_w = 800

dataset_path = './data/CULane'
row_anchor = 'culane_row_anchor'
dataset = dict(
    train=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='list/train_gt.txt'
    ),
    val=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='list/test.txt'
    ),
    test=dict(
        type='LaneClsDataset',
        img_path=dataset_path,
        data_list='list/test.txt'
    )
)

workers = 12
ignore_label = 255
log_interval = 100
eval_ep = epochs 
save_ep = epochs // 4
y_pixel_gap = 20
