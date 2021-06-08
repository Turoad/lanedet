net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet101',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
    in_channels=[64, 128, 256, 512]
)

featuremap_out_channel = 128
featuremap_out_stride = 8
sample_y = range(589, 230, -20)

batch_size = 1
aggregator = None 

neck=dict(
    type='TransConvFPN',
    in_channels=[256, 512, 1024, 256],
    out_channels=64,
    num_outs=4,
    trans_idx=-1,
    trans_cfg=dict(
        in_dim=2048,
        attn_in_dims=[2048, 256],
        attn_out_dims=[256, 256],
        strides=[1, 1],
        ratios=[4, 4],
        pos_shape=(batch_size, 10, 25),
    ),
)

num_lane_classes=1
heads=dict(
    type='CondLaneHead',
    heads=dict(hm=num_lane_classes),
    in_channels=(64, ),
    num_classes=num_lane_classes,
    head_channels=64,
    head_layers=1,
    disable_coords=False,
    branch_in_channels=64,
    branch_channels=64,
    branch_out_channels=64,
    reg_branch_channels=64,
    branch_num_conv=1,
    hm_idx=2,
    mask_idx=0,
    compute_locations_pre=True,
    location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0')
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
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)

img_height = 320 
img_width = 800
cut_height = 270 
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

