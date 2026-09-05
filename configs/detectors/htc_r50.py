"""
HTC R-50 fine-tuned on hybrid waste COCO (10 unified classes).

Base: mmdet::htc/htc-without-semantic_r50_fpn_1x_coco.py
  - Hybrid Task Cascade, 3 cascade stages (bbox + mask heads)
  - Semantic branch dropped (no pixel-wise stuff masks in the dataset)
"""

_base_ = ['mmdet::htc/htc-without-semantic_r50_fpn_1x_coco.py']

num_classes = 10

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=num_classes,
                 in_channels=256, fc_out_channels=1024, roi_feat_size=7),
            dict(type='Shared2FCBBoxHead', num_classes=num_classes,
                 in_channels=256, fc_out_channels=1024, roi_feat_size=7),
            dict(type='Shared2FCBBoxHead', num_classes=num_classes,
                 in_channels=256, fc_out_channels=1024, roi_feat_size=7),
        ],
        mask_head=[
            dict(type='HTCMaskHead', num_classes=num_classes,
                 num_convs=4, in_channels=256, conv_out_channels=256),
            dict(type='HTCMaskHead', num_classes=num_classes,
                 num_convs=4, in_channels=256, conv_out_channels=256),
            dict(type='HTCMaskHead', num_classes=num_classes,
                 num_convs=4, in_channels=256, conv_out_channels=256),
        ],
    ),
)

# ---------------------------------------------------------------------------
# Dataset — hybrid waste COCO.
# ---------------------------------------------------------------------------
data_root = 'datasets/processed_v3.3/'

metainfo = dict(
    classes=(
        'plastic_bottle', 'glass_bottle', 'glass_jar', 'metal_can',
        'carton', 'tetrapack', 'paper', 'plastic_bag',
        'plastic_container', 'other',
    ),
    palette=[
        (220,  20,  60), (119,  11,  32), (  0,   0, 142), (  0,   0, 230),
        (106,   0, 228), (  0,  60, 100), (  0,  80, 100), (  0,   0,  70),
        (  0,   0, 192), (250, 170,  30),
    ],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
    ),
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'],
)

# ---------------------------------------------------------------------------
# Training — fine-tune from COCO-pretrained HTC R-50 weights.
# The semantic branch weights in the checkpoint are silently ignored since
# our model has no semantic_head.
# ---------------------------------------------------------------------------
load_from = 'checkpoints/htc_r50_fpn_1x_coco_20200317-7332cf16.pth'

optim_wrapper = dict(
    optimizer=dict(lr=0.0025),
    clip_grad=dict(max_norm=35, norm_type=2),
)

max_epochs = 24
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
         milestones=[16, 22], gamma=0.1),
]
