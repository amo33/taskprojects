import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import random
import torch
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
from PIL import Image
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import Transform
import matplotlib.pyplot as plt
from pathlib import Path
from imantics import Polygons, Mask
import json 
import pandas as pd 
save_path = './dataset/K-Fashion/labelprocess' # 좌표 데이터 가공된 건 여기로 
df = pd.read_csv('./dataset/candiadate.csv',sep=' ')
img_number = df['file_path'].tolist()
img_style = df['스타일'].tolist()

def check_poly(data, cloth_position, arr):
    location = arr[0].tolist()
    if cloth_position == '상의':
      if data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0] != {}:
        del data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]
        data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'] = [{}]
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]['좌표'] = []
      print("상의", location)
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]['좌표'].extend(location)
      print("상의",data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['상의'][0]['좌표'] )
    elif cloth_position == '하의':
      if data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0] != {}:
        del data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]
        data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'] = [{}] 
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]['좌표'] = []
      print("하의", location)
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]['좌표'].extend(location)
      print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['하의'][0]['좌표'])
    elif cloth_position == '아우터':
      if data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0] != {}:
        del data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0]
        data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'] = [{}] 
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0]['좌표'] = []
      print("아우터: ",location)
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0]['좌표'].extend(location)
      print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['아우터'][0]['좌표'])
    elif cloth_position == '원피스':
      if data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0] != {}:
        del data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]
        data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'] = [{}] 
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]['좌표'] = []
      print("원피스", location)
      data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]['좌표'].extend(location)
      print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']['원피스'][0]['좌표'])

def check_bbox(data, cloth_position, arr):

    location = np.array(arr).reshape(1,-1)
    if cloth_position == '상의':
      if data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0] != {}:
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['X좌표'] = 0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['Y좌표'] =0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['가로'] = 0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['세로'] = 0
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['X좌표'] = arr[0]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['Y좌표'] = arr[1]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['가로'] = arr[2]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0]['세로'] = arr[3]
    elif cloth_position == '하의':
      if data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0] != {}:
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['X좌표'] = 0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['Y좌표'] = 0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['가로'] = 0
        data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['세로'] =0
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['X좌표'] = arr[0]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['Y좌표'] = arr[1]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['가로'] = arr[2]
      data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]['세로'] = arr[3]

def check_info(data):
  category_lst = ['상의','하의','원피스','아우터']
  for category in category_lst:
      if data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][category][0] != {}:
        del data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][category][0]
        data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][category] = [{}] 

setup_logger()
cfg = get_cfg()


cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "./output/model_50_0523.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
register_coco_instances("deepfashion_train", {}, "./dataset7000/train/deepfashion2_7000.json", "./dataset7000/train/image")
cfg.DATASETS.TEST = ("deepfashion_train", )

lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
            'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
            'long_sleeved_dress', 'vest_dress', 'sling_dress'] 

top = ['short_sleeved_shirt', 'long_sleeved_shirt', 'vest', 'sling']
bottom = ['shorts', 'trousers', 'skirt']
outwear = ['short_sleeved_outwear', 'long_sleeved_outwear']
onepiece = ['short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

for k in range(len(img_number)):

  img = cv2.imread('./dataset/K-Fashion/imagedata/'+img_style[k] + '/' + str(img_number[k])+".jpg") #임의로 넣어준것

  predictor = DefaultPredictor(cfg)
  outputs = predictor(img)
  # 1. lst_name안의 값들 분류해서 상의, 하의, 등등의 카테고리에 넣을 수 있는 조건문


  #print('전체 : ',outputs["instances"].pred_classes.cpu().numpy())
  json_path ='./dataset/K-Fashion/jsondata/'+img_style[k] + '/' + str(img_number[k])
  with open(json_path+'.json','r')  as json_file:
      data =json.load(json_file)
      check_info(data)
      cntNum = outputs["instances"].pred_classes.cpu().numpy().shape[0]
      targets = outputs["instances"].pred_classes.cpu().numpy()
      mask_array = outputs['instances'].pred_masks.cpu().permute(1,2,0).numpy()
      # 처음에 폴리곤 좌표 다 초기화시키는 코드를 추가해야할거 같다. ! 아래 코드 변경 시도!!!
      
      print(cntNum)
      for i in range(cntNum):
        polygon_data = Mask(mask_array[:,:,i]).polygons() # 데이터.point는 항상 1개
        #print(len(polygon_data.points))
        x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].cpu().numpy()
        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)
        w = x_max - x_min 
        h = y_max - y_min 
        bbox = [x_min, y_min, w, h]
        if lst_name[targets[i]] in top:
          key_lst = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['상의'][0].keys())
          if '카테고리' not in key_lst:
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['상의'][0]["카테고리"] = ' '
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['상의'][0]["카테고리"] = lst_name[targets[i]]
          check_poly(data, '상의' , polygon_data.points)
          check_bbox(data, '상의' , bbox)
              #print(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0])
              #print(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'])
          #data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['상의'][0] = polygon_data
        elif lst_name[targets[i]] in bottom:
          key_lst = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['하의'][0].keys())
          if '카테고리' not in  key_lst:     
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['하의'][0]["카테고리"] = ' '
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['하의'][0]["카테고리"] = lst_name[targets[i]]
          check_poly(data, '하의' , polygon_data.points)
          check_bbox(data, '하의' , bbox)
        elif lst_name[targets[i]] in outwear:
          key_lst = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['아우터'][0].keys())
          if '카테고리' not in  key_lst:     
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['아우터'][0]["카테고리"] = ' '
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['아우터'][0]["카테고리"] = lst_name[targets[i]]
          check_poly(data, '아우터' , polygon_data.points)
          check_bbox(data, '아우터' , bbox)
        elif lst_name[targets[i]] in onepiece:
          key_lst = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['원피스'][0].keys())
          if '카테고리' not in  key_lst:     
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['원피스'][0]["카테고리"] = ' '
              data['데이터셋 정보']['데이터셋 상세설명']['라벨링']['원피스'][0]["카테고리"] = lst_name[targets[i]]
          check_poly(data,'원피스' , polygon_data.points)
          check_bbox(data, '원피스' , bbox)
          #data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0] = polygon_data
        #print(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의'][0]) {~~~~~}
        #print(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표']['하의']) [{~~~~~₩}]

  with open(save_path+'/'+ str(img_number[k]) +'.json','w',encoding='UTF-8-sig') as file:
    json.dump(data, file,ensure_ascii=False)


'''
if num_instances > 0:
    for i in range(num_instances):
        im = Image.fromarray(img)
        x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].cpu().numpy()
        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)
        print(x_min)
        print(x_max)
        print(y_min)
        print(y_max)
        crop_img = im.crop((x_min, y_min, x_max, y_max))
        print("코롭 이미지")
        print(crop_img)
        ################## 해당 for문에서 json에 정보들 넣어주면 될거 같습니다 .

        #mask_array_temp = mask_array[i][y_min:y_max, x_min:x_max]
        #mask_array_temp = mask_array_temp.reshape((1,) + mask_array_temp.shape)
        #mask_array_temp = np.moveaxis(mask_array_temp, 0, -1)
        #mask_array_temp = np.repeat(mask_array_temp, 3, axis=2)
        #width = mask_array_temp.shape[0]
        #height = mask_array_temp.shape[1]
        #output = np.where(mask_array_temp==False, 255, crop_img)

    print("done")

else:
    print("nothing detected")
'''
