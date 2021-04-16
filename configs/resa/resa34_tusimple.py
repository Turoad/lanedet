net = dict(
    type='Segmentor',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
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
    dict(type='BUSD'),
    dict(type='ExistHead'),
]

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='Tusimple',        
)

optimizer = dict(
  type = 'sgd',
  lr = 0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 100
batch_size = 8
total_iter = (3616 // batch_size + 1) * epochs 
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 368
img_width = 640
cut_height = 160
seg_label = "seg_label6"

dataset_path = './data/tusimple'
dataset = dict(
    train=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='train_val_gt.txt'
    ),
    val=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    ),
    test=dict(
        type='TuSimple',
        img_path=dataset_path,
        data_list='test_gt.txt'
    )
)


workers = 12
num_classes = 6 + 1
ignore_label = 255
log_interval = 100
eval_ep = 1
save_ep = epochs
test_json_file='data/tusimple/test_label.json'
