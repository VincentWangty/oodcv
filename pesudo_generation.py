import pickle
from pascal_voc_writer import Writer
import csv
from PIL import Image
import os
import json
from collections import defaultdict

CATEGORIES = {
    0: 'aeroplane', 1: 'bicycle', 2: 'boat', 3: 'bus', 4: 'car',
    5: 'chair', 6: 'diningtable', 7: 'motorbike', 8: 'sofa', 9: 'train'
}
out_dir = '/raid/czn/oodcv/phase2-det/'
output_dir = out_dir + 'Annotations/'
json_dir = r'/raid/czn/oodcv/oodcv_tools/det_ref_0.005/merge_7x/results.json'
image_dir = r'/raid/czn/oodcv/phase2-det/JPEGImages/'
with open(json_dir, 'r', encoding='utf-8') as f:
    row_data = json.load(f)
length = len(row_data)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sign = defaultdict(list)
for i in range(length):
    name = row_data[i]['image_id']
    path = image_dir + name + '.jpg'
    if name not in sign.keys():
        sign[name].append([int(row_data[i]['bbox'][0]), \
                     int(row_data[i]['bbox'][1]), \
                     int(row_data[i]['bbox'][0]) + int(row_data[i]['bbox'][2]),
                    int(row_data[i]['bbox'][1]) + int(row_data[i]['bbox'][3]),
                      int(row_data[i]['category_id']-1),
                        float(row_data[i]['score'])
        ])
    else:
        sign[name].append([int(row_data[i]['bbox'][0]), \
                     int(row_data[i]['bbox'][1]), \
                     int(row_data[i]['bbox'][0]) + int(row_data[i]['bbox'][2]),
                    int(row_data[i]['bbox'][1]) + int(row_data[i]['bbox'][3]),
                      int(row_data[i]['category_id']-1),
                        float(row_data[i]['score'])
        ])
# print(sign)
count = 0
for key, value in sign.items():
    flag = False
    path = image_dir + key + '.jpg'
    img = Image.open(path)
    img_width = img.width  # 图片宽度
    img_height = img.height  # 图片高度
    writer = Writer(path, img_width, img_height)
    for k in range(len(value)):
        if value[k][5] > 0.7:
            writer.addObject(CATEGORIES[value[k][4]], value[k][0], value[k][1], value[k][2], value[k][3])
            flag = True
        if flag == False and k==len(value)-1:
            count += 1
    save_path = output_dir + key + '.xml'
print(count)
    # writer.save(save_path)