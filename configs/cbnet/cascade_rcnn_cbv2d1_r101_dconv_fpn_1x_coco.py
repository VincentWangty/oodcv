_base_ = '../dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py'

model = dict(
    backbone=dict(
        type='CBResNet',
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048],
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='CBFPN',
    )
)