_base_ = ['../dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py', '../_base_/swa.py']

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)