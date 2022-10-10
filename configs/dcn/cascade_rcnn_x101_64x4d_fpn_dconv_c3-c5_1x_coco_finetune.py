_base_ = ['../cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_finetune.py']
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
