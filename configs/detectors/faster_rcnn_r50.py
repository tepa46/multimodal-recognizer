"""
Faster R-CNN R-50 FPN fine-tuned on hybrid waste COCO (10 unified classes).

Base: mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
  - Bbox-only detector; no mask head.
"""

_base_ = ['mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py']

num_classes = 10

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
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
    metric=['bbox'],
)
test_evaluator = dict(
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox'],
)

load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

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
