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
import sys
import re
import time
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
import base64
import glob

app = Flask(__name__)

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

def check_element(df2, number, info):
    if number == 3:
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) &(df.ArrayFeature[1] == info[1]) & (df.ArrayFeature[2] == info[2])
        ))
    elif number == 2 or number == 1: 
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) | (df.ArrayFeature[1] == info[1]) | (df.ArrayFeature[2] == info[2])
        ))
    return df2 
#### 바로 아래부터 131번까지의 코드는 warmup을 위한 코드입니다.
df = spark.read.json("test_with_style.json")
startTime = time.time()
result_2 = [("모던",'팬츠',"우븐", None, df.collect()[-1][2]+1)] # dummy datum
extract_lst(result_2)

def extract_lst(result_2):
    global df
    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일')\
            .drop("feature",'소재')
    number = 3

    while df_spark.count() < 30:
        df_spark=check_element(df_spark, number, result_2[0])
        number -= 1 
        if number <1:
            break 
    df_spark=df_spark.limit(25) # 25개만 뽑기

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
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot)) for row in result.collect()] # 카테고리 -> category로 명명 변경 
    print("[DONE] print model_time spent :{:.4f}".format(time.time() - startTime))
    return dot_array
@app.route('/')
def main():
    return render_template('main.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/result', methods=['POST'])
def result():
    startTime = time.time()
    os.popen('docker-compose down')
    path = "/docker_file/deep_lounge_S/result"
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
    os.popen('docker-compose up -d').read() # compose 실행 백그라운드 실행 
    # sys.stdout.flush()
    log = os.popen('docker-compose logs -f').read() 
    print(log)
    style =  ["TRADITIONAL", "MANISH", "FEMININE", "ETHNIC", "CONTEMPORARY", "NATURAL", "GENDERLESS", "SPORTY", "SUBCULTURE", "CASUAL"]
    texture = ["패딩", "무스탕", "퍼프", "네오프렌", "코듀로이", "트위드", "자카드", "니트", "페플럼", "레이스", "스판덱스", "메시", "비닐/PVC", "데님", 
        "울/캐시미어", "저지", "시퀸/글리터", "퍼", "헤어 니트", "실크", "린넨", "플리스", "시폰", "스웨이드", "가죽", "우븐", "벨벳"]
    detail = ["스터드", "드롭숄더", "드롭웨이스트", "레이스업", "슬릿", "프릴", "단추", "퀄팅", "스팽글", "롤업", "니트꽈베기", "체인", 
        "프린지", "지퍼", "태슬", "띠", "플레어", "싱글브레스티드", "더블브레스티드", "스트링", "자수", "폼폼", "디스트로이드", "페플럼", 
        "X스트랩", "스티치", "레이스", "퍼프", "비즈", "컷아웃", "버클", "포켓", "러플", "글리터", "퍼트리밍", "플리츠", "비대칭", "셔링", "패치워크", "리본"]
    # print(style)
    s_data = []
    t_data = []
    d_data = []
    for data in re.findall(r"'(.*?)'", log) : 
        if data in texture : 
            t_data.append(data)
        elif data in style :
            s_data.append(data)
        elif data in detail : 
            d_data.append(data)

    for index, data in enumerate(s_data) : 
        result[index].append(data)
    for index, data in enumerate(t_data) : 
        result[index].append(data)
    for index, data in enumerate(d_data) : 
        result[index].append(data)
    print(result)
    print("[DONE] print time spent :{:.4f}".format(time.time() - startTime))
    
    return render_template('detection.html', result = result)


@app.route('/recommendation', methods=['GET'])
def redirect_page():
    
    return render_template('recommendation.html')

@app.route('/recommendation/images', methods=['GET','POST'])
def recommendation():

    startTime = time.time()
    
    result_2 = [("모던",'팬츠',"우븐", None, df.collect()[-1][2]+1)] # 제나한테서 받아야하는건 앞에 3개 포맷 만들기
    
    dot_array =extract_lst(result_2)
    startTime = time.time()
 
    recommended = sorted(dot_array, key=lambda x: x[2], reverse=True)    
    print("[DONE] print model1_time spent :{:.4f}".format(time.time() - startTime))
    startTime = time.time()
    top_list = []
    top_category = ['니트웨어','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
    for options in filter(lambda u: u[1] in top_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):
            top_list.append([options[0], options[2]])
            del top_category[top_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(top_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    #print(top_list) #확인용 
    bottom_list = []
    bottom_category = ['팬츠','스커트','청바지','레깅스','조거팬츠']
    for options in filter(lambda u: u[1] in bottom_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):

            bottom_list.append([options[0], options[2]])
            del bottom_category[bottom_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(bottom_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    #print(bottom_list) # 확인용
    candidate = []
    candidate.extend(top_list)
    candidate.extend(bottom_list)
    # top list = 상의 , bottom_list =하의 리스트 입니다. 
    # candidate는 추천 세트 후보들로 위에 2개 리스트에서 높은거부터 뽑아서 넣어줍니다.
    # !!!!! 이거 생각해보자. -> top list 와 bottom_list를 같이 합쳐서 진행해도 괜찮을까? 괜찮을거 같다. 어차피 골고루 뽑아서 재조합시키는게 나을거 같다.
    
    top_path ='static/top/' 
    bottom_path = 'static/bottom'
    result_path = 'static/recommendation' 
    #for path_name in [top_path, bottom_path, result_path]: # 이전에 사용된 이미지들은 모두 삭제
    #    [os.remove(f) for f in glob.glob(path_name+'/*')]
   
    for i in range(len(candidate)):
        file_name = candidate[i][0]
        style = candidate[i][1]
        with open('/docker_file/deepfashion/dataset/K-Fashion/labelprocess/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
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
                mask_images(box_data,key, territory, file_name,style )
    

    img_base64 = {}
    img_base64['top'] = []
    img_base64['bottom'] = []
    img_base64['recommendation'] = []
    top_lst = os.listdir(top_path)
    encodeimages(img_base64,'top',top_lst)
    bottom_lst = os.listdir(bottom_path)
    encodeimages(img_base64,'bottom',bottom_lst)
    result = get_recommendation_set(top_lst, bottom_lst) # 마지막 결과 출력 
    df = df.dropna(how='any')
    #df.drop(df.tail(1).index,inplace = True) #마지막 현 사용자에 대한 정보 없애기 
    #print(result)
    print("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
    if result != []:
        encodeimages(img_base64, 'recommendation',result)
        return render_template('recommendation.html',data = img_base64)
    else:
        return render_template('recommendation.html',data = 'error')

def get_recommendation_set(top_lst, bottom_lst):
    result_path = 'static/recommendation'
    top_path ='static/top/' # 경로 맞춰줘야한다.
    bottom_path = 'static/bottom'

    for top in top_lst:
        top_img = cv2.imread(top_path+'/'+top)
        for bottom in bottom_lst:
            #print(top)
            #print(bottom)
            bottom_img = cv2.imread(bottom_path+'/'+bottom)
            top_img = cv2.resize(top_img, (bottom_img.shape[1],top_img.shape[0]))
            result_test_concat = np.vstack([top_img, bottom_img])
            res =np.where(result_test_concat==False,255, result_test_concat)
            cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", res)
    print(os.listdir(result_path)[:5])
    return os.listdir(result_path)[:5] # 실제 이미지 경로들을 보내줘야한다. 이미지는 앞에 root 경로로 static/recommendation

def mask_images(boxes,key,territory,file_name, style):      
    #마스킹 된 걸 한번에 저장하고 싶으면 이 위치에 넣어줘야한다.(밑에 두 줄 코드)
    #img = cv2.imread("1542.jpg")
    #im = np.zeros(img.shape, dtype = np.uint8)
    img = cv2.imread("/docker_file/deepfashion/dataset/K-Fashion/imagedata/"+str(style)+'/' + str(file_name) + ".jpg")
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
            
def encodeimages(img_dict,cloth,lst):

    for i in range(len(lst)):
        with open('static/'+cloth+'/'+lst[i],'rb') as img:
            base64_image = base64.b64encode(img.read())
            img_dict[cloth].append(base64_image)


'''

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
import sys
import re
import time
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
import base64
import glob

app = Flask(__name__)

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

def check_element(df2, number, info,df ):
    if number == 3:
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) &(df.ArrayFeature[1] == info[1]) & (df.ArrayFeature[2] == info[2])
        ))
    elif number == 2 or number == 1: 
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) | (df.ArrayFeature[1] == info[1]) | (df.ArrayFeature[2] == info[2])
        ))
    return df2 
#### 바로 아래부터 131번까지의 코드는 warmup을 위한 코드입니다.
df_woman = spark.read.json("test_with_style.json")
df_man = spark.read.json("test_with_style.json")
startTime = time.time()
result_2 = [("모던",'팬츠',"우븐", None, df_woman.collect()[-1][2]+1)] # dummy datum

def extract_top_lst(result_2,df):
    startTime = time.time()
 

    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일')\
            .drop("feature",'소재')
    number = 3
    
    while df_spark.count() < 20:
        df_spark=check_element(df_spark, number, result_2[0], df)
        number -= 1 
        if number <1:
            break 
    df_spark=df_spark.limit(10) # 25개만 뽑기

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
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot)) for row in result.collect()] # 카테고리 -> category로 명명 변경 
    print("[DONE] print Dot_top_time spent :{:.4f}".format(time.time() - startTime))

    recommended = sorted(dot_array, key=lambda x: x[2], reverse=True)    
    startTime = time.time()
    top_list = []
    top_category = ['니트웨어','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
    for options in filter(lambda u: u[1] in top_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):
            top_list.append([options[0], options[2]])
            del top_category[top_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(top_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    return top_list

def extract_bottom_lst(result_2,df): # df = 남성껀지 여성껀지 구별하기 위해 input값으로 필요합니다.
    startTime = time.time()


    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일')\
            .drop("feature",'소재')
    number = 3
    
    while df_spark.count() < 20:
        df_spark=check_element(df_spark, number, result_2[0], df)
        number -= 1 
        if number <1:
            break 

    df_spark=df_spark.limit(10) # 25개만 뽑기

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
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot)) for row in result.collect()] # 카테고리 -> category로 명명 변경 
    print("[DONE] print model_time spent :{:.4f}".format(time.time() - startTime))
    bottom_list = []
    bottom_category = ['팬츠','스커트','청바지','레깅스','조거팬츠']
    recommended = sorted(dot_array, key=lambda x: x[2], reverse=True)
    for options in filter(lambda u: u[1] in bottom_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):

            bottom_list.append([options[0], options[2]])
            del bottom_category[bottom_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(bottom_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    return bottom_list

extract_top_lst(result_2, df_woman)
extract_bottom_lst(result_2, df_woman)

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/result', methods=['POST'])
def result():
    startTime = time.time()
    os.popen('docker-compose down')
    path = "/docker_file/deep_lounge_S/result"
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
    os.popen('docker-compose up -d').read() # compose 실행 백그라운드 실행 
    # sys.stdout.flush()
    log = os.popen('docker-compose logs -f').read() 
    print(log)
    style =  ["TRADITIONAL", "MANISH", "FEMININE", "ETHNIC", "CONTEMPORARY", "NATURAL", "GENDERLESS", "SPORTY", "SUBCULTURE", "CASUAL"]
    texture = ["패딩", "무스탕", "퍼프", "네오프렌", "코듀로이", "트위드", "자카드", "니트", "페플럼", "레이스", "스판덱스", "메시", "비닐/PVC", "데님", 
        "울/캐시미어", "저지", "시퀸/글리터", "퍼", "헤어 니트", "실크", "린넨", "플리스", "시폰", "스웨이드", "가죽", "우븐", "벨벳"]
    detail = ["스터드", "드롭숄더", "드롭웨이스트", "레이스업", "슬릿", "프릴", "단추", "퀄팅", "스팽글", "롤업", "니트꽈베기", "체인", 
        "프린지", "지퍼", "태슬", "띠", "플레어", "싱글브레스티드", "더블브레스티드", "스트링", "자수", "폼폼", "디스트로이드", "페플럼", 
        "X스트랩", "스티치", "레이스", "퍼프", "비즈", "컷아웃", "버클", "포켓", "러플", "글리터", "퍼트리밍", "플리츠", "비대칭", "셔링", "패치워크", "리본"]
    # print(style)
    s_data = []
    t_data = []
    d_data = []
    for data in re.findall(r"'(.*?)'", log) : 
        if data in texture : 
            t_data.append(data)
        elif data in style :
            s_data.append(data)
        elif data in detail : 
            d_data.append(data)

    for index, data in enumerate(s_data) : 
        result[index].append(data)
    for index, data in enumerate(t_data) : 
        result[index].append(data)
    for index, data in enumerate(d_data) : 
        result[index].append(data)
    print(result)
    print("[DONE] print time spent :{:.4f}".format(time.time() - startTime))
    
    return render_template('detection.html', result = result)


@app.route('/recommendation', methods=['GET'])
def redirect_page():
    
    return render_template('recommendation.html')

@app.route('/recommendation/image', methods=['GET','POST'])
def recommendation():
    # 입력받는건 최대한 2개의 배열 형태의 정보들 result_2와 같은 형태로 내가 만든다. + 남성인지 여성인지도 추가된거를 넣어주어야한다. 
    # 만약 넘어오는 값이 없다면 
    global df_woman
    global df_man
    
    result_2 = [("모던",'팬츠',"우븐", None, df_woman.collect()[-1][2]+1)] # 제나한테서 받아야하는건 앞에 3개 포맷 만들기
    
    top_list =extract_top_lst(result_2,df_woman)
    bottom_list = extract_bottom_lst(result_2,df_woman)
    startTime = time.time()
 
    #print(bottom_list) # 확인용
    candidate = []
    candidate.extend(top_list)
    candidate.extend(bottom_list)
    # top list = 상의 , bottom_list =하의 리스트 입니다. 
    # candidate는 추천 세트 후보들로 위에 2개 리스트에서 높은거부터 뽑아서 넣어줍니다.
    # !!!!! 이거 생각해보자. -> top list 와 bottom_list를 같이 합쳐서 진행해도 괜찮을까? 괜찮을거 같다. 어차피 골고루 뽑아서 재조합시키는게 나을거 같다.
    
    top_path ='static/top/' 
    bottom_path = 'static/bottom'
    result_path = 'static/recommendation' 
    #for path_name in [top_path, bottom_path, result_path]: # 이전에 사용된 이미지들은 모두 삭제
    #    [os.remove(f) for f in glob.glob(path_name+'/*')]
   
    for i in range(len(candidate)):
        file_name = candidate[i][0]
        style = candidate[i][1]
        with open('/docker_file/deepfashion/dataset/K-Fashion/labelprocess/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
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
                mask_images(box_data,key, territory, file_name,style )
    
    img_base64 = {} # img_base64 dictionary ----------- top 
    img_base64['top'] = []                  #---------- bottom  
    img_base64['bottom'] = []               #---------- recommendation # 이 3개는 각 5개의 이미지가 top1 top2 와 같은 형식의 key와 그에 따른 item(base64 이미지) 존재
    img_base64['recommendation'] = []
    top_lst = os.listdir(top_path)
    encodeimages(img_base64,'top',top_lst)
    bottom_lst = os.listdir(bottom_path)
    encodeimages(img_base64,'bottom',bottom_lst)
    result = get_recommendation_set(top_lst, bottom_lst) # 마지막 결과 출력 
    df_woman = df_woman.dropna(how='any') # 데이터 중 현 사용자 값 삭제 (file_path를 NaN으로 줘서 이 방식을 구현함)
    df_man = df_man.dropna(how='any') # 데이터 중 현 사용자 값 삭제 
    #df.drop(df.tail(1).index,inplace = True) #마지막 현 사용자에 대한 정보 없애기 
    #print(result)
    print("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
    if result != []:
        encodeimages(img_base64, 'recommendation',result)
        #print(img_base64)
        #return render_template('recommendation.html',data = img_base64)
        return json.dump(img_base64) 
    else:
        return render_template('recommendation.html',data = 'error')

def get_recommendation_set(top_lst, bottom_lst):
    result_path = 'static/recommendation'
    top_path ='static/top/' # 경로 맞춰줘야한다.
    bottom_path = 'static/bottom'

    for top in top_lst:
        top_img = cv2.imread(top_path+'/'+top)
        for bottom in bottom_lst:
            #print(top)
            #print(bottom)
            bottom_img = cv2.imread(bottom_path+'/'+bottom)
            top_img = cv2.resize(top_img, (bottom_img.shape[1],top_img.shape[0]))
            result_test_concat = np.vstack([top_img, bottom_img])
            res =np.where(result_test_concat==False,255, result_test_concat)
            cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", res)
    #print(os.listdir(result_path)[:5])
    return os.listdir(result_path)[:5] # 실제 이미지 경로들을 보내줘야한다. 이미지는 앞에 root 경로로 static/recommendation

def mask_images(boxes,key,territory,file_name, style):      
    #마스킹 된 걸 한번에 저장하고 싶으면 이 위치에 넣어줘야한다.(밑에 두 줄 코드)
    #img = cv2.imread("1542.jpg")
    #im = np.zeros(img.shape, dtype = np.uint8)
    img = cv2.imread("/docker_file/deepfashion/dataset/K-Fashion/imagedata/"+str(style)+'/' + str(file_name) + ".jpg")
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
            
def encodeimages(img_dict,cloth,lst):

    for i in range(0,len(lst)):
        data = dict()
        data['cloth'+str(i+1)] = []
        with open('static/'+cloth+'/'+lst[i],'rb') as img:
            
            base64_image = base64.b64encode(img.read())
            data['cloth'+str(i)] = base64_image
            img_dict[cloth].append(data)


'''