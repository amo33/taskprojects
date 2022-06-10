from builtins import print
from flask import Flask, render_template, request, redirect, url_for
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import random
import torch
from flask_fun import *
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
import docker
import re
import time
import datetime
import pyspark 
from pyspark.sql.functions import monotonically_increasing_id 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit
import pyspark.sql.functions as F
from pyspark.sql.functions import split, col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Normalizer
from pyspark.sql.types import StructType,StructField,ArrayType, StringType, LongType
import json 
from pyspark.sql.types import DoubleType
import uuid
import glob
import base64
from pytz import timezone
import redis
import sys


app = Flask(__name__)

r = redis.Redis()
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "../deepfashion/output/model_50_0523.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
register_coco_instances("deepfashion_train", {}, "../deepfashion/dataset7000/train/deepfashion2_7000.json", "../deepfashion/dataset7000/train/image")
cfg.DATASETS.TEST = ("deepfashion_train", )


top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"trousers"] 

spark = SparkSession\
        .builder\
        .appName('Python Spark basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()


spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
spark.conf.set("spark.python.worker.memory", '24g')

deptSchema = StructType([       
    StructField('스타일', StringType(), True),
    StructField('카테고리', StringType(), True),
    StructField('소재', StringType(), True),
    StructField('file_path', StringType(), True),
    StructField('id', LongType(), True)
])



df_woman = spark.read.json("female_style.json")
df_man = spark.read.json("male_style.json")
startTime = time.time()

result_2 = [("모던",'스커트',"우븐", None, df_woman.collect()[-1][2]+1)] # dummy datum



extract_top_lst(result_2, df_woman, 'Femaledata')
extract_bottom_lst(result_2, df_man, 'Maledata')

@app.route('/')
def main():
    return render_template('main.html')

# ali : 공유하기_메일 전송 기능
@app.route('/share')
def send_mail():
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mail')
    print('!@!@!@!@EWFQFQE')

    # 받는 사람, 받는 사진
    dest = "ali.hyung@triplllet.com"
    recommendation_path = 'static/recommendation'

    # 로그 기록
    now_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(now_time)
    now_time = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S') 

    print(dest)
    print(recommendation_path)
    print(now_time)
    os.makedirs('static/sendMaillogs', exist_ok=True)
    
    return render_template('main.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/result', methods=['POST'])
def result():
    startTime = time.time()
    # os.popen('docker-compose down')
    path = "/deep_lounge_S/deep_lounge_S/result"
    file_list = os.listdir(path)

    if os.path.exists("result") : 
        for file in os.scandir("result") : 
            os.remove(file.path)

    data = request.files['poto'].read()
    jpg_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    mask_array = outputs['instances'].pred_masks.detach().cpu().numpy()
    mask_array_temp = np.array(0)
    num_instances = mask_array.shape[0]
    # 디테일을 추가로 넣어야한다.
    result = []
    lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
                'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
                'long_sleeved_dress', 'vest_dress', 'sling_dress'] 
    top = ['short_sleeved_shirt', 'long_sleeved_shirt','short_sleeved_outwear','long_sleeved_outwear','vest',
        'sling', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']
    botton = ['shorts', 'trousers', 'skirt']
    for data in outputs["instances"].pred_classes.detach().cpu().numpy().tolist(): 
        if lst_name[data] in top : 
            result.append(['상의'])
        if lst_name[data] in botton : 
            result.append(['하의'])

    if num_instances > 0:
        for i in range(num_instances):
            im = Image.fromarray(img)
            x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].detach().cpu().numpy()
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
            output = np.where(mask_array_temp==False, 255, crop_img)
            img_process = output
            img_process = Image.fromarray(img_process)
            img_process = img_process.convert("RGBA") # 뒷배경 투명을 위해 RGBA로 엽니다. 
            datas = img_process.getdata()
            result[i].append((int((x_min+x_max)/2),int((y_min+y_max)/2)))

            newData = []
            for item in datas:
                # if item[0] > 200 and item[1] > 200 and item[2] > 200:
                #     newData.append((item[0], item[1], item[2], 0)) # 뒤에꺼는 0으로 넣어줍니다. (결과는 포토샵처럼 나옵니다.)
                if item[0] > 255 and item[1] > 255 and item[2] > 255:
                    newData.append((item[0], item[1], item[2], 0)) # 뒤에꺼는 0으로 넣어줍니다. (결과는 포토샵처럼 나옵니다.)
                else:
                    newData.append(item)

            img_process.putdata(newData)

            img_process.save("result/masked"+ str(i) +".png") # result폴더에 하나씩 만들었습니다.
    print("[DONE] print model1_time spent :{:.4f}".format(time.time() - startTime))
    print(result)
    print("==============================")
    startTime = time.time()
    # os.popen('docker-compose up -d').read() # compose 실행 백그라운드 실행 
    # sys.stdout.flush()
    print(r.get("detail"))
    # log = os.popen('docker-compose logs -f').read() 
    # style =  ["TRADITIONAL", "MANISH", "FEMININE", "ETHNIC", "CONTEMPORARY", "NATURAL", "GENDERLESS", "SPORTY", "SUBCULTURE", "CASUAL"]
    # texture = ["패딩", "무스탕", "퍼프", "네오프렌", "코듀로이", "트위드", "자카드", "니트", "페플럼", "레이스", "스판덱스", "메시", "비닐/PVC", "데님", 
    #     "울/캐시미어", "저지", "시퀸/글리터", "퍼", "헤어 니트", "실크", "린넨", "플리스", "시폰", "스웨이드", "가죽", "우븐", "벨벳"]
    # detail = ["스터드", "드롭숄더", "드롭웨이스트", "레이스업", "슬릿", "프릴", "단추", "퀄팅", "스팽글", "롤업", "니트꽈베기", "체인", 
    #     "프린지", "지퍼", "태슬", "띠", "플레어", "싱글브레스티드", "더블브레스티드", "스트링", "자수", "폼폼", "디스트로이드", "페플럼", 
    #     "X스트랩", "스티치", "레이스", "퍼프", "비즈", "컷아웃", "버클", "포켓", "러플", "글리터", "퍼트리밍", "플리츠", "비대칭", "셔링", "패치워크", "리본"]
    # # print(style)
    # s_data = []
    # t_data = []
    # d_data = []
    # for data in re.findall(r"'(.*?)'", log) : 
    #     if data in texture : 
    #         t_data.append(data)
    #     elif data in style :
    #         s_data.append(data)
    #     elif data in detail : 
    #         d_data.append(data)

    # for index, data in enumerate(s_data) : 
    #     result[index].append(data)
    # for index, data in enumerate(t_data) : 
    #     result[index].append(data)
    # for index, data in enumerate(d_data) : 
    #     result[index].append(data)
    # print(result)
    print("[DONE] print time spent :{:.4f}".format(time.time() - startTime))


    return render_template('detection.html')

@app.route('/recommendation', methods=['GET'])
def redirect_page():

    return render_template('recommendation.html')

@app.route('/recommendation/image', methods=['GET','POST'])
def recommendation():
    global df_woman
    global df_man
    params = request.get_json() 
    gender = params['gender']
    key_lst = list(params.keys())
    result_top = []
    result_bottom = []
    top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
    bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"긴바지"]
    for key in key_lst:
        temp = []
        if key_lst in top_category:
            temp.append(key)
            for top_key in list(params[key].keys()):
                temp.append(params[key][top_key])
            if gender == 'male':
                temp.append(int(df_man.collect()[-1][2])+1)
            else: 
                temp.append(int(df_woman.collect()[-1][2])+1)
            result_top.append(tuple(temp))
        elif key_lst in bottom_category:
            temp.append(key)
            for top_key in list(params[key].keys()):
                temp.append(params[key][top_key])
            if gender == 'male':
                temp.append(int(df_man.collect()[-1][2])+1)
            else: 
                temp.append(int(df_woman.collect()[-1][2])+1)
            result_bottom.append(tuple(temp))
    #result_bottom = [('모던',"조거팬츠","저지", None, int(df_woman.collect()[-1][2])+1)]
    #result_top = [('모던',"아우터","저지", None, int(df_woman.collect()[-1][2])+2)]
    if gender == "Male":
        gender = 'Maledata'
        top_list, top_info =extract_top_lst(result_top,df_man, "Maledata")
        bottom_list, bottom_info = extract_bottom_lst(result_bottom,df_man, "Maledata")
    elif gender == "Female":
        gender = 'Femaledata'
        if result_top != []:
            top_list, top_info = extract_top_lst(result_top, df_woman,"Femaledata")
        if result_bottom != []:
            bottom_list, bottom_info = extract_bottom_lst(result_bottom, df_woman, "Femaledata")
    else: 
        print("No input of gender") 
    startTime = time.time()
    #print(top_list)
    #print(bottom_list)
    #print(bottom_list) # 확인용
    candidate = []
    candidate.extend(top_list)
    candidate.extend(bottom_list)
    # top list = 상의 , bottom_list =하의 리스트 입니다.
    # candidate는 추천 세트 후보들로 위에 2개 리스트에서 높은거부터 뽑아서 넣어줍니다.
    # !!!!! 이거 생각해보자. -> top list 와 bottom_list를 같이 합쳐서 진행해도 괜찮을까? 괜찮을거 같다. 어차피 골고루 뽑아서 재조합시키는게 나을거 같다.

    top_path ='static/top'
    bottom_path = 'static/bottom'
    result_path = 'static/recommendation'
    for path_name in [top_path, bottom_path, result_path]: # 이전에 사용된 이미지들은 모두 삭제
        [os.remove(f) for f in glob.glob(os.path.join(path_name,'*'))]
    print("===")
    print(os.listdir(top_path))
    print(os.listdir(bottom_path))
    print(os.listdir(result_path))
    print("===")
    # 이미지 경로를 생각해야한다.
    for i in range(len(candidate)):
        file_name = candidate[i][0]
        style = candidate[i][1]
        with open('../deepfashion/dataset/'+gender+'/label/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
            print(file_name, style)
            data =json.load(json_file)
            box_data = {}
            territory = {}
            key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())

            key = []
            for i in range(1, len(key_data)):
                if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
                    #print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'])
                    if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]):
                        key.append(key_data[i])
                        box_data[key_data[i]] = []
                        box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]][0]['좌표'])
                        territory[key_data[i]] = []
                        territory[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][key_data[i]])
            if (box_data != {}):
                mask_images(box_data,key, territory, file_name,style, gender)

    img_base64 = {} # img_base64 dictionary ----------- top
    img_base64['top'] = []                  #---------- bottom
    img_base64['bottom'] = []               #---------- recommendation # 이 3개는 각 5개의 이미지가 top1 top2 와 같은 형식의 key와 그에 따른 item(base64 이미지) 존재
    img_base64['recommendation'] = []
    top_lst = os.listdir(top_path)
    encodeimages(img_base64,'top',top_lst)
    bottom_lst = os.listdir(bottom_path)
    encodeimages(img_base64,'bottom',bottom_lst)
    bottom_flag = [0 for _ in  range(len(bottom_list))]
    for i in range(len(bottom_flag)):
        if bottom_info[i][1] == '블라우스' or bottom_info[i][1] == '블라우스':
            bottom_flag[i] = 1
    result = get_recommendation_set(top_lst, bottom_lst,bottom_flag) # 마지막 결과 출력
    df_woman = df_woman.dropna(how='any') # 데이터 중 현 사용자 값 삭제 (file_path를 NaN으로 줘서 이 방식을 구현함)
    df_man = df_man.dropna(how='any') # 데이터 중 현 사용자 값 삭제
    #df.drop(df.tail(1).index,inplace = True) #마지막 현 사용자에 대한 정보 없애기
    #print(result)
    print("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
    if result != []:
        encodeimages(img_base64, 'recommendation',result,top_info, bottom_info)
        print(img_base64)
        #img_base64.decode('utf-8')
        #print(img_base64)
        #return render_template('recommendation.html',data = img_base64)
        return json.dumps(img_base64)
    else:
        #return render_template('recommendation.html',data = 'error')
        return json.dumps({"result":"error occured"})
