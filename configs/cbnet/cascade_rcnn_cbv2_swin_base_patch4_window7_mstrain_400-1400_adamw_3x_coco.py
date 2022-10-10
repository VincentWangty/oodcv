_base_ = [
    '../swin/cascade_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
]

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN',
    ),
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms'),
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from HTC
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='Resize',
    #     img_scale=[(1600, 400), (1600, 1400)],
    #     multiscale_mode='range',
    #     keep_ratio=True),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='PhotoMetricDistortion'),
    dict(type='OCP', prob=1, num=1, OCP=True,
         json_path='/raid/czn/oodcv/COCO_jpgjpeg/annotations/instances_trainval_new.json',
         img_path='/raid/czn/oodcv/COCO_jpgjpeg/trainval/'),
    dict(type='MixUp', prob=0.5, lambd=0.8, mixup=True,
         json_path='/raid/czn/oodcv/COCO_jpgjpeg/annotations/instances_trainval_new.json',
         img_path='/raid/czn/oodcv/COCO_jpgjpeg/trainval/'),
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CutOut', n_holes=3, cutout_ratio=(0.3, 0.3), p=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1400),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
samples_per_gpu=1
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
optimizer = dict(lr=0.0001*(samples_per_gpu/2))
