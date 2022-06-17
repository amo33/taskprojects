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
import random
import logging
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)
top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건'] # top_category, bottom_category 정의 
bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"긴바지"]
spark = SparkSession\
        .builder\
        .appName('Python Spark basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()
# spark 선언 (flask의 app 선언과 같은 아이입니다. )


spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1) # pyspark 설정 
spark.conf.set("spark.python.worker.memory", '24g') # pyspark 설정 

# pyspark에서 불러오는 데이터들은 아래와 같은 field를 가지고 있어야합니다. 다른 field들을 사용한다고 하면, 모든 input들은 각각의 field에 대한 값들을 갖고 있어야합니다.
deptSchema = StructType([ 
    StructField('file_path', StringType(), True),
    StructField('file_style_path', StringType(), True), 
    StructField('id', LongType(), True), 
    StructField('디테일', StringType(), True),
    StructField('소재', StringType(), True),
    StructField('스타일', StringType(), True),
    StructField('카테고리', StringType(), True), 
])

#result_bottom = [(None,int(df_man.collect()[-1][3])+1,'트래디셔널,서브컬쳐,캐주얼', "조거팬츠","저지", '테스트','트래디셔널')]
#result_top = [(None,'트래디셔널',int(df_man.collect()[-1][3])+1,'트래디셔널,서브컬쳐,캐주얼', '테스트',"저지", "셔츠")]
# 위 2 tuple은 dummy data

def check_element(df2, number, info,df, category): # 이 함수는 추천하기 위해 저장된 pyspark 데이터프레임 중 사용자의 데이터와 제일 유사한 데이터들만 뽑는 코드입니다. 
    # pyspark로 불러온 데이터에서 category가 같은 상의 종류, 혹은 하의 종류를 가져온다. 그리고 2순위로 스타일이 비슷한 걸 추출한다. 마지막 3순위로는 style중에 아무거나 맞는게 있다면 가져온다.
    if number == 3:
        df2 = df2.unionAll(df.filter(df.category.isin(category)))
    if number == 2:
        df2 = df2.unionAll(df.filter(
            ((df.styleArray[0] == info.head().styleArray[0]) | (df.styleArray[1] == info.head().styleArray[1]))
        ))
    elif  number == 1:
        df2 = df2.unionAll(df.filter(
            (df.styleArray[0] == info.head().styleArray[0]) | (df.styleArray[1] == info.head().styleArray[1]) | (df.file_style_path == info.head().file_style_path)
        ))

    return df2

def extract_top_lst(user_input,df, gender):
    # 상의 추출 코드입니다. 
    if len(user_input[0]) == 0: # detect 되지 않았다면 바로 exit 해준다.
        return [], []
    # [(None,'트래디셔널',-1, '슬릿',"저지",'트래디셔널,서브컬쳐,캐주얼', "셔츠")] <- dummy data 
    startTime = time.time()
    top_category = ['니트웨어','셔츠','블라우스','베스트','점퍼','티셔츠','브라탑','후드티','가디건', '긴팔 셔츠', '반팔 셔츠','긴팔 외투','반팔 셔츠','반팔 외투','긴팔 아우터','반팔 아우터']
    rdd = spark.sparkContext.parallelize(user_input)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path",'file_style_path',"id",'디테일','소재','스타일','카테고리')\
            .drop("feature")
    df_spark = df_spark.select(split(col("스타일"),",").alias("styleArray"), "ArrayFeature","file_path",'file_style_path',"id",'디테일','소재','스타일',col('카테고리').alias('category'))
    df = df.select(split(col("스타일"),",").alias("styleArray") ,"ArrayFeature","file_path",'file_style_path',"id",'디테일','소재','스타일',col('카테고리').alias('category'))
    
    # 제일 유사한 경우만 추출하는데 최대 3번 탐색할 예정
    # df_spark는 처음에 사용자 입력만 있었습니다. 이후에 df로부터 유사한 이미지 후보들을 뽑아서 추가로 넣어주는 과정을 check_element를 통해서 진행합니다 
    number = 3
    while df_spark.count() < 100:
        df_spark=check_element(df_spark, number, df_spark, df, top_category)
        number -= 1
        if number <1:
            break

    # 아래 과정은 추천 알고리즘에 쓰인 벡터화 과정 
    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf") # ArrayFeature라는 column의 값들을 tf 적용후, tf라는 column으로 저장합니다. 
    tf = hashingTF.transform(df_spark) # 맨 앞에 사용자 추가한 df_spark가 사용된다. 

    idf = IDF(inputCol="tf", outputCol="feature").fit(tf) # tf 결과를 idf화 시키는 과정을 위해 idf라는 IDF를 선언, 결과 column은 feature로 지정
    tfidf = idf.transform(tf) # idf 적용후 tfidf에 할당 

    normalizer = Normalizer(inputCol="feature", outputCol="norm") # tfidf 과정을 거친 후, 정규화 과정을 위해 Normalizer사용 
    data = normalizer.transform(tfidf) #정규화 거치고 결과는 data에 저장 

    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType()) # 사용자 정의 함수 dot_udf를 선언 (pyspark에는 없어서 만들었습니다.)
    val = user_input[0][2] # 고유 id로 사용자 id 추출

    result = data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"),
            F.col("j.id").alias("j"),
            F.col("j.file_path").alias("path"),
            F.col("j.category"),
            F.col("j.스타일").alias('style'),
            F.col("j.file_style_path"),
            F.col('j.디테일').alias('detail'),
            F.col('j.소재').alias('texture'),
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    # result는 데이터의 내적 (유사도 측정) 후의 pyspark형태입니다. 이제 result내의 row들 중 필요한 값들만 뽑아서 저장합니다 (dot_array에 저장시킵니다.)
    dot_array = [(row.path , row.category ,row.style,float(row.dot), row.file_style_path,row.detail, row.texture) for row in result.collect()] 
    logger.info("[DONE] #print Dot_top_time spent :{:.4f}".format(time.time() - startTime))
    # recommended는 유사도가 가장 높은 값부터 정렬해준다. 
    recommended = sorted(dot_array, key=lambda x: x[3], reverse=True) # 내적 결과 (유사도 값)가 제일 높은거부터 내림차순으로 정렬합니다.
  

    startTime = time.time()
    top_list = []
    top_info= []

    count = [0 for _ in range(len(top_category))]
    for options in  recommended:
  
        if os.path.exists("../deepfashion/dataset/"+ gender +'/label/'+ options[0]+".json"):
            if options[1] in top_category:
                top_list.append([options[0], options[4]])
                top_info.append(list(options))
                count[top_category.index(options[1])] +=1
                
                if count[top_category.index(options[1])] >5:
                    del top_category[top_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게
        if len(top_list) >= 10: # 단점 한 카테고리가 무더기로 나오는 현상이 발생
            break
    return top_list, top_info

def extract_bottom_lst(user_input,df, gender):
    if len(user_input[0]) == 0: # detect 되지 않았다면 바로 exit 해준다.
        return [], []
    startTime = time.time()
    bottom_category = ['팬츠','스커트','청바지','레깅스','긴바지','드레스','조거팬츠','긴팔 드레스','반팔 드레스','치마']
    rdd = spark.sparkContext.parallelize(user_input)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path",'file_style_path',"id",'디테일','소재','스타일','카테고리')\
            .drop("feature")
    df_spark = df_spark.select(split(col("스타일"),",").alias("styleArray"), "ArrayFeature","file_path",'file_style_path',"id",'디테일','소재','스타일',col('카테고리').alias('category'))
    df = df.select(split(col("스타일"),",").alias("styleArray") ,"ArrayFeature","file_path",'file_style_path',"id",'디테일','소재','스타일',col('카테고리').alias('category'))
    
    number = 3
    #하의는 60개까지만 뽑습니다. 하의 추출과정이 더 오래 걸리는 거 같아서 상의가 100개 뽑힐때 더 적게 뽑았습니다.
    while df_spark.count() < 60:
        df_spark=check_element(df_spark, number, df_spark, df, bottom_category)
        number -= 1
        if number <1:
            break

    # 아래 과정은 추천 알고리즘에 쓰인 벡터화 과정 
    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf") # ArrayFeature라는 column의 값들을 tf 적용후, tf라는 column으로 저장합니다. 
    tf = hashingTF.transform(df_spark) # 맨 앞에 사용자 추가한 df_spark가 사용된다. 

    idf = IDF(inputCol="tf", outputCol="feature").fit(tf) # tf 결과를 idf화 시키는 과정을 위해 idf라는 IDF를 선언, 결과 column은 feature로 지정
    tfidf = idf.transform(tf) # idf 적용후 tfidf에 할당 

    normalizer = Normalizer(inputCol="feature", outputCol="norm") # tfidf 과정을 거친 후, 정규화 과정을 위해 Normalizer사용 
    data = normalizer.transform(tfidf) #정규화 거치고 결과는 data에 저장 

    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType()) # 사용자 정의 함수 dot_udf를 선언 (pyspark에는 없어서 만들었습니다.)
    val = user_input[0][2] # 고유 id로 사용자 id 추출

    result = data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"),
            F.col("j.id").alias("j"),
            F.col("j.file_path").alias("path"),
            F.col("j.category"),
            F.col("j.스타일").alias('style'),
            F.col("j.file_style_path"),
            F.col('j.디테일').alias('detail'),
            F.col('j.소재').alias('texture'),
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    # result는 데이터의 내적 (유사도 측정) 후의 pyspark형태입니다. 이제 result내의 row들 중 필요한 값들만 뽑아서 저장합니다 (dot_array에 저장시킵니다.)
    dot_array = [(row.path , row.category ,row.style,float(row.dot), row.file_style_path,row.detail, row.texture) for row in result.collect()]
    logger.info("[DONE] #print model_time spent :{:.4f}".format(time.time() - startTime))
    bottom_list = []
    bottom_info = []
    
    count = [0 for _ in range(len(bottom_category))]
    recommended = sorted(dot_array, key=lambda x: x[3], reverse=True)

    for options in recommended:

        if os.path.exists("../deepfashion/dataset/"+ gender +'/label/'+ options[0]+".json"):
            if options[1] in bottom_category:
                bottom_info.append(list(options))
                bottom_list.append([options[0], options[4]])
                logger.info(options[1])
                count[bottom_category.index(options[1])] += 1
                if count[bottom_category.index(options[1])] > 4:
                    del bottom_category[bottom_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게
        if len(bottom_list) >= 10: # 단점 한 카테고리가 무더기로 나오는 현상이 발생
            break
    return bottom_list, bottom_info

def encodeimages(img_dict,cloth,lst,top_info, bottom_info, gender, input_val):
    
    # 해당 함수는 리스트형을 반환, 각 상의 하의 이미지의 flask static/옷종류/고유번호를 프론트에 보여주기 위해 반환한다. 
    # 그리고 각 이미지에 대한 정보를 보내준다. (현재 첫번째 버전은 옷의 스타일, 카테고리를 넘겨준다.)
    root_path = '175.209.155.106:5007/'

    for i in range(0,len(lst)):
        if cloth != 'recommend':
            data = dict()

        img_path = 'static/'+cloth+'/'+lst[i]
        if cloth =='recommend':

            img_dict[cloth].append(root_path+img_path)
            continue
        else:
            data['image'] = root_path + img_path
            data['context'] = "Notfound"
            data['title'] = "Notfound"
        if cloth == 'top' and top_info != []:
            data['context'] , data['title'] = extract_info(cloth,lst[i],top_info, bottom_info, gender)
        elif cloth == 'bottom' and bottom_info != []:
            data['context'] , data['title']  = extract_info(cloth,lst[i],top_info, bottom_info, gender)
        if data['context'] == "Notfound" or data['title'] == "Notfound":
            if len(input_val[0]) > 4:
                data['context'] , data['title']= input_val[0][5], input_val[0][6]
            else:
                data['context'] , data['title'] = '모던, 컨템포러리', '의류'
        img_dict[cloth].append(data)
    
    def extract_info(location,path, top_info, bottom_info, gender):
        #logger.info(top_info)
        #logger.info(bottom_info)
        info = top_info.extend(bottom_info)
        value_lst = np.array(info)
        # 정보 추출이 목적이다. 해당 함수는 원래 성별에 따른 다른 데이터셋 형태를 가지고 있어서 만든 방식이다. 하지만 박람회 출품 이틀 전에 같은 데이터셋으로부터 가져왔기에 추후에 이 코드는 줄여서 표현할 수 있습니다. 
        top_path_lst = [i[0] for i in top_info]
        bottom_path_lst = [i[0] for i in bottom_info]
        image_num_lst = []
        image_num_lst.extend(top_path_lst)
        image_num_lst.extend(bottom_path_lst)
        
        logger.info(value_lst)
        try: 
            idx = image_num_lst.index(path[:-4])
            logger.info(value_lst[idx])
        except:
            return "Notfound", "Notfound"
        
        return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail


        # if len(bottom_info) != 0:
        #     path_lst = [i[0] for i in bottom_info]
        #     if path_lst != []:
        #         #print(path_lst)
        #         value_lst = np.array(bottom_info)
        #         try: 
        #             idx = path_lst.index(path[:-4])
        #             logger.info(value_lst[idx])
        #         except:
        #             return "Notfound", "Notfound"
        #         #return "notfound","notfound"
        #         return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
    
        # if location == 'top':
        #     #logger.info("Path", path[:-4])
        #     if len(top_info) != 0:

        #         path_lst = [i[0] for i in top_info]
        

        #         if path_lst != []:

        #             value_lst = np.array(top_info)
            
        #             try: 
        #                 idx = path_lst.index(path[:-4])

        #             except:
        #                 return "Notfound", "Notfound"
        #             return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
    
        # if location == 'bottom':
        #     #print("Path", path[:-4])
        #     if len(bottom_info) != 0:
        #         path_lst = [i[0] for i in bottom_info]
        #         if path_lst != []:

        #             value_lst = np.array(bottom_info)
        #             try: 
        #                 idx = path_lst.index(path[:-4])

        #             except:
        #                 return "Notfound", "Notfound"
        #             #return "notfound","notfound"
        #             return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
            


# def extract_info(location,path, top_info, bottom_info, gender):
#     logger.info(top_info)
#     logger.info(bottom_info)
#     # 정보 추출이 목적이다. 해당 함수는 원래 성별에 따른 다른 데이터셋 형태를 가지고 있어서 만든 방식이다. 하지만 박람회 출품 이틀 전에 같은 데이터셋으로부터 가져왔기에 추후에 이 코드는 줄여서 표현할 수 있습니다. 
#     if gender == 'Femaledata':
#         if location == 'top':
#             if len(top_info) != 0:
#                 path_lst = [i[0] for i in top_info]
        
#                 if path_lst != []:

#                     value_lst = np.array(top_info)
#                     logger.info(value_lst)
#                     try: 
#                         idx = path_lst.index(path[:-4])
#                         logger.info(value_lst[idx])
#                     except:
#                         return "Notfound", "Notfound"
                    
#                     return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
    
#         if location == 'bottom':
#             if len(bottom_info) != 0:
#                 path_lst = [i[0] for i in bottom_info]
#                 if path_lst != []:
#                     #print(path_lst)
#                     value_lst = np.array(bottom_info)
#                     try: 
#                         idx = path_lst.index(path[:-4])
#                         logger.info(value_lst[idx])
#                     except:
#                         return "Notfound", "Notfound"
#                     #return "notfound","notfound"
#                     return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
#     if gender == 'Maledata':
#         if location == 'top':
#             #logger.info("Path", path[:-4])
#             if len(top_info) != 0:

#                 path_lst = [i[0] for i in top_info]
        

#                 if path_lst != []:

#                     value_lst = np.array(top_info)
              
#                     try: 
#                         idx = path_lst.index(path[:-4])
  
#                     except:
#                         return "Notfound", "Notfound"
#                     return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
    
#         if location == 'bottom':
#             #print("Path", path[:-4])
#             if len(bottom_info) != 0:
#                 path_lst = [i[0] for i in bottom_info]
#                 if path_lst != []:

#                     value_lst = np.array(bottom_info)
#                     try: 
#                         idx = path_lst.index(path[:-4])

#                     except:
#                         return "Notfound", "Notfound"
#                     #return "notfound","notfound"
#                     return convertstyle(value_lst[idx][2], 'kor'), value_lst[idx][1] #, value_lst[idx][6], value_lst[idx][5] #순서 : style, category, texture, detail
              

def convertstyle(stylestring,language):
    # 해당 함수는 한글과 영어로 들어온 스타일을 모델 inference 과정에 필요한 형태에 맞추기 위해 사용됨 styles는 서비스 후반에 갑자기 영어를 생각하지 않아도 되는 걸로 변경되자 모든 코드를 바꾸기에는 시간이 부족해서 쓴 방법

    stylekor =  {'레트로':'RETRO','로맨틱':'ROMANTIC',"리조트":"RESORT","모던":"MODERN","밀리터리":"MILITARY","섹시":"SEXY","소피스티케이티드":"SOPHISTICATED","스트리트":"STREET","아방가르드":"AVANGARD","오리엔탈":"ORIENTAL","클래식":"CLASSIC","키치":"KICHI","펑크":"PUNK","프레피":"PREP","히피":"HIPPIE","힙합":"HIPHOP","트래디셔널":"TRADITIONAL", '매니쉬':"MANISH", "페미닌":"FEMININE", "에스닉":"ETHNIC", "컨템포러리":"CONTEMPORARY", "내츄럴":"NATURAL", '젠더리스':"GENDERLESS", "스포티":"SPORTY", "서브컬쳐":"SUBCULTURE", "캐주얼":"CASUAL"}
    style_lst = [x.strip() for x in stylestring.split(',')]
    for i in style_lst[:]:
        if i == '':
            style_lst.remove(i)
    logger.info(style_lst)
    styleeng = {'RETRO':'레트로','ROMANTIC':'로맨틱',"RESORT":"리조트","MODERN":"모던","MILITARY":"밀리터리","SEXY":"섹시","SOPHISTICATED":"소피스티케이티드","STREET":"스트리트","AVANGARD":"아방가르드","ORIENTAL":"오리엔탈","CLASSIC":"클래식","KICHI":"키치","PUNK":"펑크","PREP":"프레피","HIPPIE":"히피","HIPHOP":"힙합","TRADITIONAL":"트래디셔널", "MANISH":"매니쉬", "FEMININE":"페미닌", "ETHNIC":"에스닉", "CONTEMPORARY":"컨템포러리", "NATURAL":"내츄럴", "GENDERLESS":"젠더리스", "SPORTY":"스포티", "SUBCULTURE":"서브컬쳐", "CASUAL":"캐주얼"}
    result_val = ''
    styles = {'레트로':'레트로','로맨틱':'로맨틱',"리조트":"리조트","모던":"모던","밀리터리":"밀리터리","섹시":"섹시","소피스티케이티드":"소피스티케이티드","스트리트":"스트리트","아방가르드":"아방가르드","오리엔탈":"오리엔탈","클래식":"클래식","키치":"키치","펑크":"펑크","프레피":"프레피","히피":"히피","힙합":"힙합","트래디셔널":"트레디셔널", '매니쉬':"매니쉬", "페미닌":"페미닌", "에스닉":"에스닉", "컨템포러리":"컨템포러리", "내츄럴":"내츄럴", '젠더리스':"젠더리스", "스포티":"스포티", "서브컬쳐":"서브컬쳐", "캐주얼":"캐주얼"}
    if language == 'kor':
        for i in range(len(style_lst)):
            #result_val += stylekor[style_lst[i]] # 원래는 stylekor 로 접근
            result_val += styles[style_lst[i]]
            if i< len(style_lst) -1:
                result_val += ','
    elif language == 'eng':
        for i in range(len(style_lst)):
            # result_val += styleeng[style_lst[i]] # 원래는 styleeng 로 접근 
            result_val += styles[style_lst[i]]
            if i< len(style_lst) -1:
                result_val += ','
    return result_val

def get_recommendation_set(top_lst, bottom_lst,bottom_flag):
    
    # 이 함수는 추천 조합을 뽑아주기 위해 사용되었습니다. 1. 상의 하의 모두 정보를 갖고 있으면 둘이 조합 // 2. 만약 옷 중, 하의가 원피스나 드레스라면 상의와 조합하지 말고 그냥 하의만 저장시킨다. 
    # 위 주석에 설명된 2번의 경우를 판별하기 위해서,  bottom_flag 리스트를 사용해서 구현했다.
    result_path = 'static/recommend'
    top_path ='static/top' # 경로 맞춰줘야한다.
    bottom_path = 'static/bottom'
    for j in range(len(top_lst)):
        top_img = cv2.imread(top_path+'/'+top_lst[j])
        if len(bottom_flag) == 0:
            res = np.where(top_img > 255 ,255, top_img)
            resize_img = cv2.resize(res, (450, 900), interpolation=cv2.INTER_AREA)
            cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", resize_img)
            continue
        for i in range(len(bottom_flag)):
        
            if bottom_flag[i] == 0:
                bottom_img = cv2.imread(bottom_path+'/'+bottom_lst[i])
                top_img = cv2.resize(top_img, (bottom_img.shape[1],top_img.shape[0]))
                result_test_concat = np.vstack([top_img, bottom_img])
                res =np.where(result_test_concat>255,255, result_test_concat)
                resize_img = cv2.resize(res, (450, 900), interpolation=cv2.INTER_AREA)
                cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", resize_img)
            elif bottom_flag[i] == 1:
                bottom_img =  cv2.imread(bottom_path+'/'+bottom_lst[i])
                res =np.where(bottom_img> 255,255, bottom_img)
                resize_img = cv2.resize(res, (450, 900), interpolation=cv2.INTER_AREA)
                cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", resize_img)

    return os.listdir(result_path) # 실제 이미지 경로들을 보내줘야한다. 이미지는 앞에 root 경로로 static/recommendation

def mask_images(boxes,key,territory,file_name, style, gender):
    # mask image 함수는 모델 착용 이미지가 들어오면 이미지 내에 상, 하의 등을 분리시켜 각각 top, bottom에 저장하는 과정이다. 이 과정에서 전체 사진 면적 중 masking 면적이 너무 적거나 비정상적인 가로 세로 비율은 무시했다.
    img = cv2.imread("../deepfashion/dataset/"+gender + "/Image/"+str(style)+'/' + str(file_name) + ".jpg")
    total_size = img.shape[0] * img.shape[1]
    for j in range(len(key)):
        dst2= []
        im = np.zeros(img.shape, dtype = np.uint8)
        channel_count = img.shape[2]

        # aspect ration로 적은 정보를 가지고 있는 이미지는 제거
        if territory[key[j]][0] == {}:
            continue
        h=np.int32( territory[key[j]][0]['세로'])
        w = np.int32( territory[key[j]][0]['가로'])
        
        if (float(w)/h < 0.45): # 이 수치는 deepfashion의 aspect ratio를 측정해 만든 것이다.
            continue

        l= (np.array(boxes[key[j]]).astype(np.int32))

        y_low = np.int32(territory[key[j]][0]['Y좌표'])
        y_high = y_low +np.int32( territory[key[j]][0]['세로'])
        x_low = np.int32(territory[key[j]][0]['X좌표'])
        x_high = x_low + np.int32(territory[key[j]][0]['가로'])

        mask = np.zeros(img.shape[:2], np.uint8)
        ctr = np.array(l).reshape((-1,1,2)).astype(np.int32)
 
        if float(cv2.contourArea(ctr))/total_size <= 0.010:
            # 만약 마스킹 면적 비율이 0.01보다 작거나 같으면 저장안하고 넘어간다.
            continue
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
            ##print(key[j])
            cv2.imwrite("static/top/"+ file_name +".png",result_img)
        else:
            ##print(key[j])
            cv2.imwrite("static/bottom/"+file_name +".png",result_img)

def checkinfo(top_lst,bottom_lst,top_info, bottom_info, gender):
    # 현재 static/top과 static/bottom 에 있는 옷들만 걸러서 생각한다.
    result_top = []
    result_bottom = []
    top_path = 'static/top/'
    bottom_path = 'static/bottom/'

    if top_lst != []:
        tops = [i for i in top_lst]
    else:
        tops = []
    if bottom_lst != []:
        bottoms = [i for i in bottom_lst]
    else:
        bottoms = []

    if top_info != []:
        path_lst = [i[0] for i in top_info]
    else:
        path_lst = []
    
    if bottom_info != []:
        path_lst2 = [i[0] for i in bottom_info]

    else:
        path_lst2 = []

    if tops != [] and path_lst !=[]:
        for top in tops[:]:
            if top[:-4] in path_lst:
                result_top.append(top_info[path_lst.index(top[:-4])])
            else:
                os.remove(top_path+top)
                tops.remove(top)
    if bottoms !=[] and path_lst2 !=[]:
        for bottom in bottoms:
            if bottom[:-4] in path_lst2:
                result_bottom.append(bottom_info[path_lst2.index(bottom[:-4])])
            else:
                os.remove(bottom_path+bottom)
                bottoms.remove(bottom)

    return tops, bottoms , result_top, result_bottom
  
# check_info func return should be expanded into 4. 
# status codes implements that this function return only 2 which does not return updated list of static files
# And then we need to check encode images function if the previous code does not work .
# below code is an backup.
'''
#{'gender': '{gender=여성}', 'bottom': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}', 'top': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}'}
	global df_woman
	global df_man
	#params = request.form.to_dict()
	#{'gender': {'gender':'여성'}, 'bottom': {'style':'캐주얼,내츄럴,컨템포러리', 'detail':'팬츠', 'category':'팬츠', 'texture':'팬츠'}, 'top': {'style':'팬츠', 'detail':'팬츠', 'category':'팬츠', 'texture':'팬츠'}}
	param = {'gender': '{gender=F}', 'bottom': '{style=GENDERLESS,CASUAL, detail=스트링, category=팬츠, texture=우븐}', 'top': '{style=매니쉬,캐주얼, detail=드롭숄더, category=셔츠, texture=니트}'}
	#{'gender': '{gender=여성}', 'bottom': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}', 'top': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}'}
	top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
	bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"긴바지"]
	params = param 
	#request.form.to_dict() 
	#print(params['gender'].split('=')[1][0])
	gender = 'Femaledata' if params['gender'].split('=')[1][0] == 'F' else 'Maledata'
	#gender = "Maledata"
	print(gender)
	#print(df_man.collect()[-1])
	result_bottom = []
	result_top = []
	if 'bottom' in params.keys():
		datum = []
		dilim = param['bottom'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		parsed_data = dilim.split("/")
		#print(re.split('{ |style= |, detail= |, category= |, texture= |}',params['bottom']))
		#print(params['bottom'][0:6])
		datum.append(None)
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(convertstyle(parsed_data[1],'eng')) # style 
		datum.append(parsed_data[3])

		#result_bottom = [(None,'트래디셔널',-1, '슬릿',"저지",'트래디셔널,캐주얼,컨템포러리', "긴바지")]
		datum = tuple(datum)
		result_bottom.append(datum)
	
	if 'top' in params.keys():
		datum = []
		dilim = param['top'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		datum.append(None)
		parsed_data = dilim.split("/")
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(parsed_data[1])
		datum.append(parsed_data[3])
		datum = tuple(datum)
		result_top.append(datum)
		#result_top = [(None,'트래디셔널',-1, '슬릿',"저지",'트래디셔널,서브컬쳐,캐주얼', "셔츠")]

	if gender == "Maledata":
		top_list, top_info =extract_top_lst(result_top,df_man, gender)
		bottom_list, bottom_info = extract_bottom_lst(result_bottom,df_man, gender)
	else:
		top_list, top_info = extract_top_lst(result_top, df_woman, gender)
		bottom_list, bottom_info = extract_bottom_lst(result_bottom, df_woman, gender) 
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
	result_path = 'static/recommend'
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
			#print(file_name, style)
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
	print("===")
	print(os.listdir(top_path))
	print(os.listdir(bottom_path))
	print(os.listdir(result_path))
	print("===")
	img_base64 = {} # img_base64 dictionary ----------- top
	img_base64['top'] = []                  #---------- bottom
	img_base64['bottom'] = []               #---------- recommendation # 이 3개는 각 5개의 이미지가 top1 top2 와 같은 형식의 key와 그에 따른 item(base64 이미지) 존재
	img_base64['recommend'] = []
	#print("top",top_info)
	#print("botoom", bottom_info)
	top_lst = os.listdir(top_path)
	bottom_lst = os.listdir(bottom_path)
	#print()
	top_lst, bottom_lst, top_info, bottom_info = checkinfo(top_lst, bottom_lst,top_info, bottom_info, gender)
	#print("top",top_info)
	#print("botoom", bottom_info)
	top_lst = os.listdir(top_path)
	encodeimages(img_base64,'top',top_lst, top_info,bottom_info, gender)
	top_lst = os.listdir(top_path)
	encodeimages(img_base64,'bottom',bottom_lst, top_info, bottom_info, gender)
	bottom_lst = os.listdir(bottom_path)
	bottom_set = set(bottom_lst) #집합set으로 변환
	bottom_lst = list(bottom_set) #list로 변환
	top_set = set(top_lst)
	top_lst = list(top_set)
	#print("Duplicate done bottom", bottom_lst)
	#print("Duplicate done top", top_lst)
	if len(bottom_lst) < len(bottom_info):
		bottom_flag = [0 for _ in  range(len(bottom_lst))]
	elif len(bottom_lst) >= len(bottom_info):
		bottom_flag = [0 for _ in  range(len(bottom_info))]
	#print("flag",bottom_flag)
	print(len(bottom_info), len(bottom_lst))
	for i in range(len(bottom_flag)):
		if bottom_info[i][1] == '드레스':
			bottom_flag[i] = 1
	result = get_recommendation_set(top_lst, bottom_lst,bottom_flag) # 마지막 결과 출력
	df_woman = df_woman.dropna(how='any') # 데이터 중 현 사용자 값 삭제 (file_path를 NaN으로 줘서 이 방식을 구현함)
	df_man = df_man.dropna(how='any') # 데이터 중 현 사용자 값 삭제
	print(result)
	print("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
	if result != []:
		encodeimages(img_base64, 'recommend',result, top_info,bottom_info, gender)
		#print(img_base64)
		return make_response(json.dumps(img_base64, ensure_ascii=False)) # json.dumps
		#return make_response(json.dumps(img_base64, ensure_ascii=False).decode('utf8'))
	else:
		return jsonify({"top":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 "bottom":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 "recommend":["img1","img2","img3"] })
	#request: Analysis + gender
	#return: recommendItemDict
	# print(request.form.to_dict())
	 #return jsonify({
	 #"top":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 #"bottom":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 #"recommend":["img1","img2","img3"] 
	 #})
'''