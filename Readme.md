## Introduction

Our code is based on the repo https://github.com/VDIGPKU/CBNetV2 and https://github.com/WongKinYiu/yolov7

+ Two-stage Framework:

  The config of Stage-1 training is common,  the config of Stage-2 finetuning is end with finetune. 

+ NS-WBF : NS-WBF/wbf_ensemble_multimodel.py

+ OCP (mmdet/datasets/pipelines/transforms.py #3097)：

  ```python
  @PIPELINES.register_module()
  class OCP(object):
  
      def __init__(self, prob=0.5, num=1, OCP=True,
                   json_path='data/coco/annotations/pgtrainval2017.json',
                   img_path='data/coco/images/'):
          self.prob = prob
          self.OCP = OCP
          self.num = num
          self.json_path = json_path
          self.img_path = img_path
          with open(json_path, 'r') as json_file:
              all_labels = json.load(json_file)
          self.all_labels = all_labels
  
      def get_img2(self):
          idx2 = np.random.choice(np.arange(len(self.all_labels['images'])))
          img2_fn = self.all_labels['images'][idx2]['file_name']
          img2_id = self.all_labels['images'][idx2]['id']
          img2_path = self.img_path + img2_fn
          img2 = cv2.imread(img2_path)
  
          # get image2 label
          labels2 = []
          boxes2 = []
          for annt in self.all_labels['annotations']:
              if annt['image_id'] == img2_id:
                  labels2.append(np.int64(annt['category_id']))
                  boxes2.append([np.float32(annt['bbox'][0]),
                                 np.float32(annt['bbox'][1]),
                                 np.float32(annt['bbox'][0] + annt['bbox'][2] - 1),
                                 np.float32(annt['bbox'][1] + annt['bbox'][3] - 1)])
          return img2, labels2, boxes2
  
      def __call__(self, results):
          if self.OCP == True:
              if random.uniform(0, 1) < self.prob:
                  # object-copy-paste
                  img1 = results['img']
                  labels1 = results['gt_labels']
                  img2, labels2, boxes2 = self.get_img2()
                  x1 = int(boxes2[0][0])
                  x2 = int(boxes2[0][2])
                  y1 = int(boxes2[0][1])
                  y2 = int(boxes2[0][3])
                  height = max(img1.shape[0], img2.shape[0])
                  width = max(img1.shape[1], img2.shape[1])
                  object = img2.astype('float32')[y1:y2,x1:x2,:]
                  object1 = copy.deepcopy(object)
                  copy_paste_image = np.zeros([height, width, 3], dtype='float32')
                  copy_paste_image[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')
                  if object1.shape != copy_paste_image[y1:y2,x1:x2,:].shape:
                      return results
                  else:
                      copy_paste_image[y1:y2,x1:x2,:] = object1
                      img_cp = copy_paste_image.astype('uint8')
  
                      # mix labels
                      results['gt_labels'] = np.hstack((labels1, np.array([labels2[0]])))
                      results['gt_bboxes'] = np.vstack((list(results['gt_bboxes']), [boxes2[0]]))
                      results['img'] = img_cp
                  return results
              else:
                  return results
  ```

