import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from glob import glob
import json
import tqdm
import numpy as np
import pdb
import cv2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file', default="/raid/czn/oodcv/CBNetV2/work_dirs/regnet_finaltest/cascade_rcnn_regnetx-12GF_fpn_1x_coco_finetune.py")
    parser.add_argument('--checkpoint', help='Checkpoint file', default="/raid/czn/oodcv/CBNetV2/work_dirs/regnet_finaltest/epoch_12.pth")
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def main(args):
    nuisances = ['weather', 'pose', 'texture', 'context', 'shape', 'occlusion', 'iid_test']
    PATH = '/raid/czn/oodcv/nuisances_p2/'
    # filename = '/raid/OOD-CV/CBNetV2/work_dirs/faster_rcnn_r50_fpn_1x_voc0712/inference/iid_test.json'
    model = init_detector(args.config, args.checkpoint)
    for i in tqdm.tqdm(range(len(nuisances))):
        final = []
        # filename = '/raid/OOD-CV-COCO/CBNetV2/work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_coco/inference_0.05_true/' + nuisances[i] +'.json'
        filename = '/raid/czn/oodcv/CBNetV2/work_dirs/regnet_finaltest/inference_all_ep12_score0.005/' + \
                   nuisances[i] + '.json'
        images = glob(PATH + nuisances[i]+'/*')
        # print(images)
        for k in range(len(images)):
            # print(images[k])
            try:
                result = inference_detector(model, images[k])
            except:
                print(images[k])
                continue
            # print(result)
            for m in range(len(result)):
                if result[m].any()>0:
                    for q in range(len(result[m])):
                        result_json = {}
                        if result[m][q][4]>0:
                # if result[m][4]>args.score_thr:
                            result_json["bbox"] = np.array([int(result[m][q][0]),int(result[m][q][1]),int(result[m][q][2]-result[m][q][0]),int(result[m][q][3]-result[m][q][1])]).tolist()
                            result_json["image_id"] = images[k].split('/')[-1].split('.')[0]
                            result_json["score"] = float(result[m][q][4])
                            # if nuisances[i] == "texture":
                            #     result_json["category_id"] = int(int(m+2) % 10)
                            # elif nuisances[i] == "weather":
                            #     result_json["category_id"] = int(int(m + 1) % 10 + 1)
                            # else:
                            #     result_json["category_id"] = int(m+1)
                            result_json["category_id"] = int(m + 1)
                            final.append(result_json)
        with open(filename, 'w') as res:
            json.dump(final, res)


if __name__ == '__main__':
    args = parse_args()
    main(args)
