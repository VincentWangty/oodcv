import argparse
from argparse import ArgumentParser
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pdb
from glob import glob
import random
import time

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', help='Checkpoint file', default="/disk2/wty/OOD-CV-COCO/yolov7/runs/train/yolov7-e6-finetune2/weights/epoch_210.pt")
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument(
        '--device', default='cuda:5', help='Device used for inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    args = parser.parse_args()
    return args

def main(args):
    nuisances = ['shape', 'pose', 'texture', 'context', 'weather', 'occlusion', 'iid_test']
    # nuisances = ['occlusion', 'iid_test']
    PATH = '/disk2/wty/OOD-CV-COCO/data/ROBINv1.1/nuisances_p2/'
    # filename = '/raid/OOD-CV/CBNetV2/work_dirs/faster_rcnn_r50_fpn_1x_voc0712/inference/iid_test.json'
    model = attempt_load(args.checkpoint, map_location=args.device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check img_size
    model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # Run inference
    if args.device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(args.device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    for i in tqdm(range(len(nuisances))):
        final = []
        # filename = '/raid/OOD-CV-COCO/CBNetV2/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco/inference_0.05_true/' + nuisances[i] +'.json'
        filename = '/disk2/wty/OOD-CV-COCO/yolov7/runs/train/yolov7-e6-finetune2/finetune_stage2/' + \
                   nuisances[i] + '.json'
        image_path = PATH + nuisances[i] + '/'
        print(image_path)
        dataset = LoadImages(image_path, img_size=imgsz, stride=stride)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(args.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if args.device != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=args.augment)[0]

            # Inference
            pred = model(img, augment=args.augment)[0]

            # Apply NMS
            result = non_max_suppression(pred, 0.05, 0.5, classes=args.classes,
                                       agnostic=args.agnostic_nms)

            # Afterwards
            for i, det in enumerate(result):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # normalized xywh
                        result_json = {}
                        result_json["bbox"] = [int(xywh[0] * gn[0].item() - (xywh[2] * gn[2].item())/2), int(xywh[1] * gn[1].item() - (xywh[3] * gn[3].item())/2), int(xywh[2] * gn[2].item()), int(xywh[3] * gn[3].item())]
                        result_json["image_id"] = path.split('/')[-1].split('.')[0]
                        result_json["score"] = float(conf.item())
                        result_json["category_id"] = int(cls.item()) + 1
                        final.append(result_json)
        with open(filename, 'w') as res:
            json.dump(final, res)

if __name__ == '__main__':
    args = parse_args()
    main(args)
