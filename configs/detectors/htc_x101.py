"""
HTC ResNeXt-101-64x4d FPN fine-tuned on hybrid waste COCO (10 unified classes).

Base: mmdet::htc/htc-without-semantic_r50_fpn_1x_coco.py (same as htc_r50.py),
  backbone swapped to ResNeXt-101-64x4d. We intentionally do NOT inherit from
  `htc_x101-64x4d_fpn_16xb1-20e_coco.py` because that base pulls in the
  semantic branch.
"""

_base_ = ['mmdet::htc/htc-without-semantic_r50_fpn_1x_coco.py']

num_classes = 10

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d'),
    ),
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
    batch_size=1,
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

load_from = 'checkpoints/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth'

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
