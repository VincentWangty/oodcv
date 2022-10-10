# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/disk2/wty/OOD-CV-COCO/data/ROBINv1.1/COCO_jpg_slice_512/'
data_root_train='/raid/czn/oodcv/COCO_jpgjpeg+test/'
data_root = '/raid/czn/oodcv/COCO_test/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.6),
]

train_pipeline = [
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
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Resize', img_scale=(500, 375), keep_ratio=True),
    # dict(
    #     type='Resize',
    #     img_scale=[(1333, 400), (1333, 1000)],
    #     multiscale_mode='range',
    #     keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CutOut', n_holes=3, cutout_ratio=(0.3, 0.3), p=1),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(1266, 760),
        # img_scale=[(1333, 600),(1333, 800),(1333, 1000),(1333, 1200)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='PhotoMetricDistortion'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_trainval_new.json',
        img_prefix=data_root + 'trainval/',
        pipeline=train_pipeline),
    # train=dict(
    #     type='MultiImageMixDataset',
    #         dataset=dict(
    #             type=dataset_type,
    #             ann_file=data_root + 'annotations/instances_trainval.json',
    #             img_prefix=data_root + 'trainval/',
    #             pipeline=[
    #                 dict(type='LoadImageFromFile'),
    #                 dict(type='LoadAnnotations', with_bbox=True),
    #             ],
    #             filter_empty_gt=True),
    #     pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_trainval_new.json',
        img_prefix=data_root + 'trainval/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_trainval_new.json',
        img_prefix=data_root + 'trainval/',
        pipeline=test_pipeline))
evaluation = dict(interval=12, metric='bbox')
