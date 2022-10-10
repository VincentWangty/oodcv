_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        # fusion_factors=[1.49, 3.03, 4.86])
        # fusion_factors=[0.135, 0.374, 0.523])
        fusion_factors=[0.206, 0.33, 0.671])
)