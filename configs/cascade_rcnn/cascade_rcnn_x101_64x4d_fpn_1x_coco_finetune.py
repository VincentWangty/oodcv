_base_ = './cascade_rcnn_r50_fpn_1x_coco_finetune.py'
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/spring/.cache/torch/hub/checkpoints/resnext101_64x4d-173b62eb.pth')))
