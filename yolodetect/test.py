# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
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
# import matplotlib as mpl

setup_logger()
cfg = get_cfg()

img = cv2.imread("testlast.jpeg")

cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model_0009999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
register_coco_instances("deepfashion_train", {}, "./dataset7000/train/deepfashion2_7000.json", "./dataset7000/train/image")
cfg.DATASETS.TEST = ("deepfashion_train", )


predictor = DefaultPredictor(cfg)
outputs = predictor(img)
#outputs = model(inputs)
print(outputs["instances"].pred_classes) 
print(outputs["instances"].pred_boxes)
print(outputs["instances"].scores)
print(outputs["instances"].pred_masks)
mask_array = outputs['instances'].pred_masks.numpy()
print(mask_array)
mask_array_temp = np.array(0)
num_instances = mask_array.shape[0]
print(num_instances)
origin_w , origin_h = img.shape[0] , img.shape[1]
if num_instances > 0:
    for i in range(num_instances):
        im = Image.fromarray(img)
        x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].numpy()
        x_min = int(x_min)
        x_max = int(x_max)
        y_max = int(y_max)
        y_min = int(y_min)
        crop_img = im.crop((x_min, y_min, x_max, y_max))
        mask_array_temp = mask_array[i][y_min:y_max, x_min:x_max]
        mask_array_temp = mask_array_temp.reshape((1,) + mask_array_temp.shape)
        mask_array_temp = np.moveaxis(mask_array_temp, 0, -1)
        mask_array_temp = np.repeat(mask_array_temp, 3, axis=2)
        width = mask_array_temp.shape[0]
        height = mask_array_temp.shape[1]
        print(width, height)
        print(origin_w ,origin_h)
        output = np.where(mask_array_temp==False, 255, crop_img)
    
        cv2.imwrite("result"+str(i)+ ".png", output)

        img_process = Image.open("result"+str(i)+ ".png")
        print(img_process.size)
        img_process = img_process.convert("RGBA") # 뒷배경 투명을 위해 RGBA로 엽니다. 
        datas = img_process.getdata()

        newData = []
        for item in datas:
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                newData.append((item[0], item[1], item[2], 0)) # 뒤에꺼는 0으로 넣어줍니다. (결과는 포토샵처럼 나옵니다.)
            else:
                newData.append(item)

        img_process.putdata(newData)

        img_process.save("result"+ str(i) +".png")
    print("done")
else:
    print("nothing detected")
