checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/disk2/wty/OOD-CV-COCO/CBNetV2/work_dirs/cascade_rcnn_x101_64x4d_fpn_dconv_c3-c5_1x_coco_lr0.02_bs8_mixup_new+cutout_new+ciou+scales8+PMD+rcnn_thr_500375_newcoco_72ep/epoch_72.pth'
load_from = '/raid/czn/oodcv/CBNetV2/work_dirs/regnetx_12GF_bs4_cutout0.5/epoch_96.pth'
resume_from = None
workflow = [('train', 1)]
