_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_finetune.py',
    '../_base_/schedules/schedule_1x_finetune.py', '../_base_/default_runtime_finetune.py'
]
