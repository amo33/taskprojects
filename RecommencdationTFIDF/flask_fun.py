from builtins import print
from flask import Flask, render_template, request, redirect, url_for
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import random
import torch

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

def check_element(df2, number, info,df):
    #result_2 = [("모던",'팬츠',"우븐", None, int(df_woman.collect()[-1][2])+1)]
    if number == 3:
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) & (df.ArrayFeature[2] == info[2])
        ))
    elif number == 2 or number == 1:
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0])  | (df.ArrayFeature[2] == info[2])
        ))
    return df2

def extract_top_lst(result_2,df, gender):
    startTime = time.time()
    #gender = 'Maledata' if df == df_man else 'Femaledata'
    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'스타일','카테고리','file_style_path')\
            .drop("feature",'소재')
    number = 3

    while df_spark.count() < 25:
        df_spark=check_element(df_spark, number, result_2[0], df)
        number -= 1
        if number <1:
            break
    df_spark=df_spark.limit(25) # 25개만 뽑기
    #print('top')
    #print(df_spark.show())
    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf")
    tf = hashingTF.transform(df_spark) # 맨 마지막 df row에 사용자 추가한 df 가 combined data
    #tf = hashingTF.transform(combined_data) # 원래 쓰던거
    idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
    tfidf = idf.transform(tf)

    normalizer = Normalizer(inputCol="feature", outputCol="norm")
    data = normalizer.transform(tfidf)

    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType())
    val = result_2[0][4] # 고유 id로 사용자 id 추출

    result = data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"),
            F.col("j.id").alias("j"),
            F.col("j.file_path").alias("path"),
            F.col("j.카테고리").alias("category"),
            F.col("j.스타일").alias('style'),
            F.col("j.file_style_path"),
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot), row.file_style_path,row.detail, row.texture) for row in result.collect()] # 카테고리 -> category로 명명 변경
    print("[DONE] print Dot_top_time spent :{:.4f}".format(time.time() - startTime))
    #print(result.show())
    #print(dot_array[3][3])
    recommended = sorted(dot_array, key=lambda x: x[3], reverse=True)
    #print(recommended[:10])
    startTime = time.time()
    top_list = []
    top_info= []
    top_category = ['니트웨어','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
    for options in filter(lambda u: u[1] in top_category , recommended):
        #print(options)
        if os.path.exists("../deepfashion/dataset/"+ gender +'/label/'+ options[0]+".json"):
            top_list.append([options[4], options[2]])
            top_info.append(options)
            del top_category[top_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게
        if len(top_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생
            break
    return top_list

def extract_bottom_lst(result_2,df, gender): # df = 남성껀지 여성껀지 구별하기 위해 input값으로 필요합니다.
    startTime = time.time()
    #print('bottom')
    #gender = 'Maledata' if df == df_man else 'Femaledata'
    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일','file_style_path')\
            .drop("feature",'소재')
    number = 3
    #print(df_spark.show())
    #print(df.show())
    while df_spark.count() < 50:
        df_spark=check_element(df_spark, number, result_2[0], df)
        number -= 1
        if number <1:
            break
    #print(df_spark)
    df_spark=df_spark.limit(40) # 25개만 뽑기
    print('bottom')
    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf")
    tf = hashingTF.transform(df_spark) # 맨 마지막 df row에 사용자 추가한 df 가 combined data
    #tf = hashingTF.transform(combined_data) # 원래 쓰던거
    idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
    tfidf = idf.transform(tf)

    normalizer = Normalizer(inputCol="feature", outputCol="norm")
    data = normalizer.transform(tfidf)

    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType())
    val = result_2[0][4] # 고유 id로 사용자 id 추출

    result = data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"),
            F.col("j.id").alias("j"),
            F.col("j.file_path").alias("path"),
            F.col("j.카테고리").alias("category"),
            F.col("j.디테일").alias("detail"),
            F.col("j.소재").alias("texture"),
            F.col("j.스타일").alias('style'),
            F.col('j.file_style_path'),
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot), row.file_style_path,row.detail, row.texture) for row in result.collect()] # 카테고리 -> category로 명명 변경
    #print(result.show())
    print("[DONE] print model_time spent :{:.4f}".format(time.time() - startTime))
    bottom_list = []
    bottom_info = []
    bottom_category = ['팬츠','스커트','청바지','레깅스','조거팬츠']
    recommended = sorted(dot_array, key=lambda x: x[3], reverse=True)
    #print("recommend",recommended[:10])
    for options in filter(lambda u: u[1] in bottom_category , recommended):
        print(options)
        if os.path.exists("../deepfashion/dataset/"+ gender +'/label/'+ options[0]+".json"):
            bottom_info.append(options)
            bottom_list.append([options[4], options[2]])
            del bottom_category[bottom_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게
        if len(bottom_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생
            break
    return bottom_list, bottom_info

def encodeimages(img_dict,cloth,lst,top_info, bottom_info):

    for i in range(0,len(lst)):
        data = dict()
        data['cloth'+str(i+1)] = []
        with open('static/'+cloth+'/'+lst[i],'rb') as img:

            base64_image = base64.b64encode(img.read())
            data['cloth'+str(i+1)] = base64_image.decode('utf8')
            data['style'] , data['category'] , data['texture'] ,data['detail'] = extract_info(cloth,lst[i],top_info, bottom_info)
            img_dict[cloth].append(data)

def extract_info(location,path, top_info, bottom_info):
    if location == 'top':
        
        path_lst = np.array(top_info).T[0]
        idx = path_lst.index(path)
        return '#'+top_info[idx][2], '#'+top_info[idx][1] , '#'+top_info[idx][6], '#'+top_info[idx][5] #순서 : style, category, texture, detail
    else:
        path_lst = np.array(bottom_info).T[0]
        idx = path_lst.index(path)
        return '#'+bottom_info[idx][2], '#'+bottom_info[idx][1], '#'+bottom_info[idx][6], '#'+bottom_info[idx][5]


def get_recommendation_set(top_lst, bottom_lst,bottom_flag):
    result_path = 'static/recommendation'
    top_path ='static/top/' # 경로 맞춰줘야한다.
    bottom_path = 'static/bottom'

    for top in top_lst:
        top_img = cv2.imread(top_path+'/'+top)
        for i in  range(len(bottom_lst)):
            #print(top)
            #print(bottom)
            if bottom_flag[i] == 0:
                bottom_img = cv2.imread(bottom_path+'/'+bottom_lst[i])
                top_img = cv2.resize(top_img, (bottom_img.shape[1],top_img.shape[0]))
                result_test_concat = np.vstack([top_img, bottom_img])
                res =np.where(result_test_concat==False,255, result_test_concat)
            else:
                bottom_img =  cv2.imread(bottom_path+'/'+bottom_lst[i])
                res =np.where(bottom_img==False,255, bottom_img)
            cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", res)
    #print(os.listdir(result_path)[:5])
    return os.listdir(result_path)[:5] # 실제 이미지 경로들을 보내줘야한다. 이미지는 앞에 root 경로로 static/recommendation

def mask_images(boxes,key,territory,file_name, style, gender):
    #마스킹 된 걸 한번에 저장하고 싶으면 이 위치에 넣어줘야한다.(밑에 두 줄 코드)
    #img = cv2.imread("1542.jpg")
    #im = np.zeros(img.shape, dtype = np.uint8)
    print(gender)
    img = cv2.imread("../deepfashion/dataset/"+gender + "/Image/"+str(style)+'/' + str(file_name) + ".jpg")
    for j in range(len(key)):
        dst2= []
        im = np.zeros(img.shape, dtype = np.uint8)
        channel_count = img.shape[2]

        # aspect ration로 적은 정보를 가지고 있는 이미지는 제거
        if territory[key[j]][0] == {}:
            continue
        h=np.int32( territory[key[j]][0]['세로'])
        w = np.int32( territory[key[j]][0]['가로'])
        if (float(w)/h < -1): # 이 수치는 deepfashion의 aspect ratio를 측정해 만든 것이다.
            continue

        l= (np.array(boxes[key[j]]).astype(np.int32))
        #print(l)
        y_low = np.int32(territory[key[j]][0]['Y좌표'])
        y_high = y_low +np.int32( territory[key[j]][0]['세로'])
        x_low = np.int32(territory[key[j]][0]['X좌표'])
        x_high = x_low + np.int32(territory[key[j]][0]['가로'])

        #croped = img[y_low:y_high, x_low:x_high].copy()

        #l = l - l.min(axis=0)

        mask = np.zeros(img.shape[:2], np.uint8)
        ctr = np.array(l).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) bit-operation 과정
        dst = cv2.bitwise_and(img, img, mask=mask)

        ## (4) white background으로 변경
        bg = np.ones_like(img, np.uint8)*255
        cv2.imwrite(str(j)+"test.png",mask)
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2.append(bg + dst)

        result_img = dst2[0][y_low:y_high, x_low:x_high]
        if key[j] in top_category:
            #print(key[j])
            cv2.imwrite("static/top/result_test"+ file_name[:4] + str(j)+".png",result_img)
        else:
            #print(key[j])
            cv2.imwrite("static/bottom/result_test"+file_name[:4] + str(j)+".png",result_img)

