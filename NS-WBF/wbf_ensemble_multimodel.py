from ensemble_boxes import *
import json
import cv2
import matplotlib.pyplot as plt
import tqdm
from collections import defaultdict
import os
from PIL import Image
import glob
import numpy as np

def get_weight(paths):
    weights = []
    models = []
    for i in range(len(paths)):
        AP50s = []
        if os.path.exists(paths[i]+'.txt'):
            with open(paths[i]+'.txt', 'r') as result:
                result = result.readlines()
                for j in range(7):
                    if j==6:
                        AP50s.append(float(result[j+1].split(':')[1].split('\\')[0]))
                    else:
                        AP50s.append(float(result[j].split(':')[1].split('\\')[0]))
        else:
            for j in range(7):
                AP50s.append(float(1/11))
        models.append(AP50s)
    for i in range(7):
        sort = np.array(models)[:, i]
        index = np.argsort(-sort)
        weight = np.zeros((len(paths)))
        for k in range(len(paths)):
            weight[index[k]] = float((len(paths)-k) * 1 / len(paths))
        weights.append(weight)
    return weights

def wbf(dets, name, iou_thr, weights, skip_box_thr, conf_type):
    boxes_list = [[] for i in range(len(dets))]
    scores_list = [[] for i in range(len(dets))]
    labels_list = [[] for i in range(len(dets))]
    for i in range(len(dets)):
        for j in range(len(dets[i])):
            boxes_list[i].append(dets[i][j][0:4])
            scores_list[i].append(dets[i][j][5])
            labels_list[i].append(dets[i][j][4])
    # 实现weight boxes fusion
    boxes, scores, labels = weighted_boxes_fusion(boxes_list,
                                                  scores_list,
                                                  labels_list,
                                                  weights=weights,
                                                  iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr,
                                                  conf_type=conf_type)
    # 还原normalized的bbox
    img_path = r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.jpg'
    if os.path.exists(img_path):
        img = Image.open(r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.jpg')
    else:
        img = Image.open(r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.JPEG')
    imgw = img.width  # 图片宽度
    imgh = img.height  # 图片高度
    new_boxes = []
    for i, box in enumerate(boxes):
        box_x1 = float(box[0] * imgw)
        box_y1 = float(box[1] * imgh)
        box_w = float((box[2] - box[0]) * imgw)
        box_h = float((box[3] - box[1]) * imgh)
        new_boxes.append([box_x1, box_y1, box_w, box_h])
    return new_boxes, list(scores), list(labels)

def transform_wbf_dets(path):
    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion', 'iid_test']
    # for path in paths:
        # filename = os.path.join(path, 'test.json')
    final = []
    for i in tqdm.tqdm(range(len(nuisances))):
        per = defaultdict(list)
        file = os.path.join(path, nuisances[i]+'.json')
        with open(file, 'r') as a:
            result = json.load(a)
            for j in range(len(result)):
                name = result[j]["image_id"]
                img_path = r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.jpg'
                if os.path.exists(img_path):
                    img = Image.open(r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.jpg')
                else:
                    img = Image.open(r'/raid/czn/oodcv/VOC2007/testimg/' + name + '.JPEG')
                img_width = img.width  # 图片宽度
                img_height = img.height  # 图片高度
                result_list = []
                result_list.append(float(result[j]["bbox"][0])/img_width)
                result_list.append(float(result[j]["bbox"][1])/img_height)
                result_list.append((float(result[j]["bbox"][0]) + float(result[j]["bbox"][2]))/img_width)
                result_list.append((float(result[j]["bbox"][1]) + float(result[j]["bbox"][3]))/img_height)
                result_list.append(result[j]["category_id"])
                result_list.append(result[j]["score"])
                per[name].append(result_list)
        final.append(per)
    return final
def transform_wtf_back(final, index):
    path = r'/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/merge_11x_new_finetune_final'
    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion', 'iid_test']
    filename = os.path.join(path, nuisances[index] + '.json')
    results = []
    for j in range(len(final)):
        for k in range(len(final[j][1][0])):
            result_json = {}
            result_json["bbox"] = np.array([int(final[j][1][0][k][0]),int(final[j][1][0][k][1]),int(final[j][1][0][k][2]),int(final[j][1][0][k][3])]).tolist()
            result_json["image_id"] = final[j][0]
            result_json["score"] = float(final[j][1][1][k])
            result_json["category_id"] = int(final[j][1][2][k])
            results.append(result_json)
    with open(filename, 'w') as res:
        json.dump(results, res)
    return

iou_thr = 0.4
# weights = [0.9, 0.6, 0.5, 0.3, 0.2]
skip_box_thr = 0.0
conf_type = 'max'
# weights = [[1, 0.8, 0.6, 0.4, 0.2], [1, 0.8, 0.6, 0.4, 0.2],[1, 0.8, 0.4, 0.6, 0.2], [1, 0.8, 0.6, 0.4, 0.2], [1, 0.8, 0.6, 0.4, 0.2],
# [0.6, 1, 0.8, 0.4, 0.2], [1, 0.2, 0.4, 0.8, 0.6]]
# model_paths = ['/disk2/wty/ROBIN-dataset/evaluation/det_ref/swa_x101_72ep_inference_all_swa1-12_score0.005',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/swa_cascade_rcnn_r101_fpn_1x_coco_inference_all_swa1-12_score0.005',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/r50_inference_all_ep72_score0.005_1333800',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/yolov7-w6_inference_ep99',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/yolov7-e6_inference_ep199',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/cascade_rcnn_x101_64x4d_fpn_dconv_c3-c5_1x_coco_lr0.02_bs8_mixup_new+cutout_new0.5+OCP+ciou+scales8+rcnn_thr_500375_newcoco_inference_all_ep96_score0.005',
#             '/disk2/wty/ROBIN-dataset/evaluation/det_ref/model_ensemble_wbf_merge3x_new'
# ]
# model_paths = ['/raid/czn/oodcv/oodcv_tools/det_ref_0.005/cascade_rcnn_x101_64x4d_fpn_dconv_c3-c5_1x_coco_lr0.02_bs8_mixup_new+cutout_new0.5+OCP+ciou+scales8+rcnn_thr_500375_newcoco_inference_all_ep96_score0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/cascade_rcnn_x101_64x4d_fpn_dconv_c3-c5_1x_coco_lr0.02_bs8_mixup_new+cutout_new0.8+OCP+ciou+scales8+rcnn_thr_500375_newcoco_ep144_inference_all_ep144_score0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/r50_inference_all_ep72_score0.005_1333800',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/regnetx_12GF_bs4_cutout0.5_inference_all_ep96_score0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/swa_cascade_rcnn_r101_fpn_1x_coco_inference_all_swa1-12_score0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/x101_OCP_wocutout_2e-3finetune_24ep_inference_all_ep24_0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/yolov7-e6_inference_ep199',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/yolov7-w6_inference_ep99',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/x101_pesudo_ep36_inference_all_ep36_score0.005',
#             '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/x101_pesudo+initial_ep36_inference_all_ep36_score0.005',
# ]
# model_paths = ['/raid/czn/oodcv/oodcv_tools/det_ref_0.005/x101_pesudo_ep36_inference_all_ep36_score0.005',
#                '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/merge_8x_stage2',
#                '/raid/czn/oodcv/oodcv_tools/det_ref_0.005/x101_pesudo+initial_ep36_inference_all_ep36_score0.005',
# ]
model_paths = ['/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/r50_pesudo_finetune_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/r50_largesize_finetune',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/r101_swa_pesudo_finetune_inference_all_ep36_score0.005',
                # '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/regnetx_pesudo_finetune_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/x101_finetune_pesudo_finetune_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/x101_pesudo_ep36_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/x101_largesize_finetune_ep36_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/detectors_r101_pesudo_finetune',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/yolo-w6_pesudo_finetune',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/s101_pesudo_finetune_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/regnet_occlusion_inference_all_ep36_score0.005',
                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/yolo-e6_pesudo_finetune']

# model_paths = ['/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/chair+results',
#                '/raid/czn/oodcv/oodcv_tools/det_ref_stage2_finetune/merge_8x_new_finetune',
# ]

weights = get_weight(model_paths)
print(weights)
# weights = [0.8, 0.9, 1, 0.6, 0.4]
# weights = [0.5 ,0.5]
new_result = []
# for i in range(len(weights)):
#     weights.append()


# model1_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/model_ensemble_wbf_x101_swa+r101_swa+r50_ori'
# model2_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/model_ensemble_wbf_r101+x101+detectors'
# model3_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/model_ensemble_wbf_r101+x101+cbnet'

# model1_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/cascade_rcnn_cbv2d1_r101_mdconv_fpn_1x_coco_lr0.01_bs2_mixup+cutout+giou+size1000600_newcoco_inference_all_ep36'
# model2_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/cascade_rcnn_x101_64x4d_fpn_dconv_c3-c5_1x_coco_lr0.02_bs8_mixup_new+cutout_new+ciou+scales8+rcnn_thr_500375_newcoco_inference_all_best_ep44'
# model3_path = '/disk2/wty/ROBIN-dataset/evaluation/det_ref/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_lr0.04_bs8_mixup_new+cutout_new+giou+scales8+rcnn_thr_500375_newcoco_inference_all_ep36'
formats = []
for num in range(len(model_paths)):
    formats.append(transform_wbf_dets(model_paths[num]))
for i in range(7):
    final = []
    for key, value in formats[0][i].items():
        dets = []
        for num in range(len(model_paths)):
            dets.append(formats[num][i][key])
        ens = wbf(dets, key, iou_thr, weights[i], skip_box_thr, conf_type)
        # ens = wbf(dets, key, iou_thr, weights, skip_box_thr, conf_type)
        final.append([key, ens])
    transform_wtf_back(final, i)