net = dict(
    type='Detector',
)

backbone = dict(
    type='MobileNet',
    net='MobileNetV2',
    pretrained=True,
    out_conv=False,
)
featuremap_out_channel = 1280
featuremap_out_stride = 32 

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -1)

heads = dict(type='LaneATT',
        anchors_freq_path='.cache/culane_anchors_freq.pt',
        topk_anchors=1000)

train_parameters = dict(
    conf_threshold=None,
    nms_thres=15.,
    nms_topk=3000
)
test_parameters = dict(
    conf_threshold=0.5,
    nms_thres=50,
    nms_topk=max_lanes
)

optimizer = dict(
  type = 'AdamW',
  lr = 0.0003,
)

epochs = 15
batch_size = 8
total_iter = (88880 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 1

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
ori_img_w=1640
ori_img_h=590
img_w=640 
img_h=360
cut_height=0

train_process = [
    dict(type='GenerateLaneLine', 
        transforms = (
            dict(
                name = 'Affine',
                parameters = dict(
                    translate_px = dict(
                        x = (-25, 25),
                        y = (-10, 10)
                    ),
                    rotate=(-6, 6),
                    scale=(0.85, 1.15)
                )
            ),
            dict(
                name = 'HorizontalFlip',
                parameters = dict(
                    p=0.5
                ),
            )
        ), 
        wh = (img_w, img_h),
    ),
    dict(type='ToTensor', keys=['img', 'lane_line']),
] 

val_process = [
    dict(type='GenerateLaneLine'),
    dict(type='ToTensor', keys=['img']),
] 

dataset_path = './data/CULane'
dataset_type = 'CULane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
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
log_interval = 1000
seed=0
lr_update_by_epoch = False
